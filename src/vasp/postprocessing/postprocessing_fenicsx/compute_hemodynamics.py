import dolfinx
import numpy as np
import numpy.typing as npt
from scifem.interpolation import SurfaceSubmeshInterpolation
from fenicsx_compat.mesh import submesh_to_parent
import ufl


class InterpolateDG:
    """Interpolate a function from a DG function space to a subspace."""

    def __init__(
        self,
        expr: dolfinx.fem.Expression,
        V_sub: dolfinx.fem.FunctionSpace,
        entity_map: dolfinx.mesh.EntityMap,
    ):
        self.V_sub = V_sub
        self.entity_map = entity_map

        assert V_sub.mesh.topology == entity_map.sub_topology
        parent_facets = submesh_to_parent(entity_map)
        exterior_facet_indices = dolfinx.mesh.exterior_facet_indices(
            entity_map.topology
        )
        sub_facets = np.arange(len(parent_facets), dtype=np.int32)
        assert np.isin(parent_facets, exterior_facet_indices).all()
        assert len(parent_facets) == len(exterior_facet_indices)
        domains = ufl.domain.extract_domains(expr)
        assert len(domains) == 1, "Expression must be defined on a single domain"
        domain = domains[0].ufl_cargo()

        assert domain.topology == entity_map.topology._cpp_object

        interpolation_entities = dolfinx.fem.compute_integration_domains(
            dolfinx.fem.IntegralType.exterior_facet,
            dolfinx.mesh.Topology(domain.topology),
            exterior_facet_indices,
        ).reshape(-1, 2)
        self._interpolator = SurfaceSubmeshInterpolation(
            expr, V_sub, sub_facets, interpolation_entities
        )

    def __call__(self, u_surface: dolfinx.fem.Function):
        self._interpolator.apply(u_surface)


class Stress:
    """Wall shear stress of a velocity field, on a facet submesh of its mesh.

    The wall shear stress is the tangential part of the traction ``F = -sigma n`` with the
    Cauchy stress ``sigma = 2 mu_f sym(grad(u)) - p I``. The pressure only contributes to the
    normal part, so it is left out. The expression is compiled once, and every call evaluates it
    at the current values of ``u`` directly at the interpolation points of ``V_sub``, without a
    projection.

    Args:
        u: Velocity field on the parent mesh
        V_sub: Vector space on the facet submesh to evaluate the wall shear stress in
        mu_f: Dynamic viscosity
        entity_map: Entity map from :py:func:`dolfinx.mesh.create_submesh` relating the submesh
            of ``V_sub`` to the mesh of ``u``
    """

    def __init__(
        self,
        u: dolfinx.fem.Function,
        V_sub: dolfinx.fem.FunctionSpace,
        mu_f: float | dolfinx.fem.Constant,
        entity_map: dolfinx.mesh.EntityMap,
    ):
        n = ufl.FacetNormal(u.function_space.mesh)
        F = -2 * mu_f * ufl.sym(ufl.grad(u)) * n
        self._interpolator = InterpolateDG(F - ufl.inner(F, n) * n, V_sub, entity_map)
        self._tau = dolfinx.fem.Function(V_sub, name="WSS")

    @property
    def tau(self) -> dolfinx.fem.Function:
        """The wall shear stress from the last call."""
        return self._tau

    def __call__(self) -> dolfinx.fem.Function:
        """Compute the wall shear stress at the current values of the velocity field."""
        self._interpolator(self._tau)
        return self._tau


def _unit_cube_boundary(cell_type: dolfinx.mesh.CellType, N: int = 4):
    """A unit cube, its exterior facets as a submesh, and the submesh's entity map."""
    from mpi4py import MPI

    domain = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, N, N, N, cell_type=cell_type)
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, fdim + 1)
    facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    submesh, entity_map = dolfinx.mesh.create_submesh(domain, fdim, facets)[:2]
    return domain, submesh, entity_map


def _unit_cube_outward_normal(submesh: dolfinx.mesh.Mesh) -> npt.NDArray[np.float64]:
    """The outward unit normal of the unit cube at each local submesh cell, from its midpoint."""
    tdim = submesh.topology.dim
    num_cells = submesh.topology.index_map(tdim).size_local
    midpoints = dolfinx.mesh.compute_midpoints(
        submesh, tdim, np.arange(num_cells, dtype=np.int32)
    )
    normals = np.zeros((num_cells, 3))
    normals[np.isclose(midpoints, 0.0)] = -1.0
    normals[np.isclose(midpoints, 1.0)] = 1.0
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    return normals


def _cell_values(u: dolfinx.fem.Function) -> npt.NDArray[np.float64]:
    """Values of a blocked DG function at the dofs of each local cell, (num_cells, num_dofs, bs)."""
    V = u.function_space
    num_cells = V.mesh.topology.index_map(V.mesh.topology.dim).size_local
    bs = V.dofmap.index_map_bs
    return u.x.array.reshape(-1, bs)[V.dofmap.list[:num_cells]]


_CELL_TYPES = (dolfinx.mesh.CellType.tetrahedron, dolfinx.mesh.CellType.hexahedron)


def test_interpolate_dg_reproduces_polynomial():
    """A degree 2 volume field is reproduced exactly by DG2 on the boundary.

    Checks that each submesh cell receives the values of its own parent facet.
    """

    def f(x):
        return (x[0] ** 2, x[1] * x[2], x[0] * x[1] - x[2])

    for cell_type in _CELL_TYPES:
        domain, submesh, entity_map = _unit_cube_boundary(cell_type)
        u = dolfinx.fem.Function(dolfinx.fem.functionspace(domain, ("DG", 2, (3,))))
        u.interpolate(f)
        V_sub = dolfinx.fem.functionspace(submesh, ("DG", 2, (3,)))
        u_sub = dolfinx.fem.Function(V_sub)
        InterpolateDG(u, V_sub, entity_map)(u_sub)

        u_exact = dolfinx.fem.Function(V_sub)
        u_exact.interpolate(f)
        np.testing.assert_allclose(u_sub.x.array, u_exact.x.array, atol=1e-12)


def test_stress():
    """The wall shear stress of a linear velocity field, which depends on the facet normal.

    With u = A x, sigma = mu (A + A^T) is constant, so on each boundary facet with outward normal
    n the tangential traction F - (F . n) n, F = -sigma n, is a constant vector. One ``Stress``
    is reused for two velocity fields, as in a time loop.
    """
    mu = 3.5
    A0 = np.array([[0.3, -1.2, 0.7], [2.0, 0.1, -0.4], [-0.8, 1.5, 0.6]])

    for cell_type in _CELL_TYPES:
        domain, submesh, entity_map = _unit_cube_boundary(cell_type)
        u = dolfinx.fem.Function(
            dolfinx.fem.functionspace(domain, ("Lagrange", 2, (3,)))
        )
        V_sub = dolfinx.fem.functionspace(submesh, ("DG", 1, (3,)))
        stress = Stress(u, V_sub, mu, entity_map)
        normals = _unit_cube_outward_normal(submesh)

        for A in (A0, -2.5 * A0.T):
            u.interpolate(lambda x: A @ x)
            tau = stress()
            assert tau is stress.tau

            sigma = mu * (A + A.T)
            F_exact = -normals @ sigma  # sigma is symmetric
            Ft_exact = F_exact - np.sum(F_exact * normals, axis=1)[:, None] * normals
            values = _cell_values(tau)
            np.testing.assert_allclose(
                values, np.broadcast_to(Ft_exact[:, None, :], values.shape), atol=1e-12
            )


if __name__ == "__main__":
    test_interpolate_dg_reproduces_polynomial()
    test_stress()
