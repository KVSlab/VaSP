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


def cauchy_stress(
    u: ufl.core.expr.Expr,
    p: ufl.core.expr.Expr | float,
    mu_f: ufl.core.expr.Expr | float,
) -> ufl.core.expr.Expr:
    """The Cauchy stress ``sigma = 2 mu_f sym(grad(u)) - p I`` of a Newtonian fluid."""
    return 2 * mu_f * ufl.sym(ufl.grad(u)) - p * ufl.Identity(u.ufl_shape[0])


def wall_shear_stress(sigma: ufl.core.expr.Expr, n: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    """The wall shear stress of a stress tensor on a fluid boundary with outward unit normal ``n``.

    The fluid exerts the traction ``F = -sigma n`` on the wall, and the wall shear stress is its
    tangential part ``(I - n n^T) F = F - (F . n) n``. An isotropic part ``-p I`` of ``sigma``
    gives ``F = p n``, which has no tangential part, so the pressure does not contribute and
    ``sigma`` may be passed with or without it.
    """
    F = -sigma * n
    return F - ufl.inner(F, n) * n


class Stress:
    """Wall shear stress of a velocity field, on a facet submesh of its mesh.

    See :py:func:`wall_shear_stress`. Since the pressure does not contribute, only the viscous
    part ``2 mu_f sym(grad(u))`` of the Cauchy stress is used. The expression is compiled once,
    and every call evaluates it at the current values of ``u`` directly at the interpolation
    points of ``V_sub``, without a projection.

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
        sigma_viscous = cauchy_stress(u, 0.0, mu_f)
        self._interpolator = InterpolateDG(
            wall_shear_stress(sigma_viscous, n), V_sub, entity_map
        )
        self._tau = dolfinx.fem.Function(V_sub, name="WSS")

    @property
    def tau(self) -> dolfinx.fem.Function:
        """The wall shear stress from the last call."""
        return self._tau

    def __call__(self) -> dolfinx.fem.Function:
        """Compute the wall shear stress at the current values of the velocity field."""
        self._interpolator(self._tau)
        return self._tau


class HemodynamicIndices:
    """Time-averaged hemodynamic indices from a sequence of wall shear stress samples.

    With ``<f> = 1 / T int_{t_0}^{t_N} f dt`` the time average over ``T = t_N - t_0``, the indices
    are (see https://kvslab.github.io/VaMPy/quantities.html)

    - ``TAWSS = <|tau|>``
    - ``OSI = (1 - |<tau>| / TAWSS) / 2``
    - ``RRT = 1 / |<tau>|``, which equals ``1 / ((1 - 2 OSI) TAWSS)``
    - ``ECAP = OSI / TAWSS``
    - ``TWSSG = <|d tau / dt|>``

    The time steps between samples may vary. ``<tau>`` and ``<|tau|>`` use the trapezoidal rule,
    and ``tau`` is taken as linear in time between samples for ``TWSSG``, which then is
    ``sum_k |tau_k - tau_{k-1}| / T``. Where the denominator of OSI, RRT or ECAP is zero (no
    wall shear stress, or none on average), that index is set to zero.

    The indices are computed dof by dof, in a scalar space with the same element and dofmap as
    each component of the wall shear stress space. For a Lagrange space, this is the
    interpolation of each (nonlinear) index into it, since its dofs are point evaluations.

    Args:
        V: Vector space of the wall shear stress samples
    """

    def __init__(self, V: dolfinx.fem.FunctionSpace):
        self._V = V
        self._bs = V.dofmap.index_map_bs
        element = V.ufl_element()
        assert len(V.value_shape) == 1, "The wall shear stress space must be vector valued"
        self._Q = dolfinx.fem.functionspace(V.mesh, element.sub_elements[0])
        assert np.array_equal(self._Q.dofmap.list, V.dofmap.list)

        num_dofs = V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts
        dtype = V.mesh.geometry.x.dtype
        # Time integrals over [t_0, t_prev] of tau, |tau| and |d tau / dt|
        self._int_tau = np.zeros((num_dofs, self._bs), dtype=dtype)
        self._int_tau_magnitude = np.zeros(num_dofs, dtype=dtype)
        self._int_twssg = np.zeros(num_dofs, dtype=dtype)
        self._tau_prev = np.zeros_like(self._int_tau)
        self._tau_magnitude_prev = np.zeros_like(self._int_tau_magnitude)
        self._t_first: float | None = None
        self._t_prev: float | None = None
        self._num_samples = 0

    @property
    def num_samples(self) -> int:
        """The number of wall shear stress samples added."""
        return self._num_samples

    def add(self, tau: dolfinx.fem.Function, t: float):
        """Add the wall shear stress sample ``tau`` at time ``t``.

        Args:
            tau: Wall shear stress in the space the indices were created with
            t: Time of the sample, larger than that of the previous sample
        """
        assert tau.function_space == self._V
        tau_k = tau.x.array.reshape(-1, self._bs)
        tau_magnitude_k = np.linalg.norm(tau_k, axis=1)
        if self._t_prev is None:
            self._t_first = t
        else:
            dt = t - self._t_prev
            assert dt > 0, "Samples must be added in increasing time"
            self._int_tau += 0.5 * dt * (self._tau_prev + tau_k)
            self._int_tau_magnitude += 0.5 * dt * (self._tau_magnitude_prev + tau_magnitude_k)
            # |d tau / dt| = |tau_k - tau_{k-1}| / dt on the interval, integrated over dt
            self._int_twssg += np.linalg.norm(tau_k - self._tau_prev, axis=1)
        self._tau_prev[:] = tau_k
        self._tau_magnitude_prev[:] = tau_magnitude_k
        self._t_prev = t
        self._num_samples += 1

    def compute(self) -> dict[str, dolfinx.fem.Function]:
        """The indices from the samples added so far, keyed and named by their abbreviation."""
        assert self._num_samples >= 2, "At least two samples are needed"
        assert self._t_first is not None and self._t_prev is not None
        T = self._t_prev - self._t_first
        tawss = self._int_tau_magnitude / T
        mean_tau_magnitude = np.linalg.norm(self._int_tau / T, axis=1)

        def divide(a: npt.NDArray[np.floating], b: npt.NDArray[np.floating]):
            """``a / b``, and zero where ``b`` is zero."""
            return np.divide(a, b, out=np.zeros_like(a), where=b > 0)

        osi = 0.5 * (1 - divide(mean_tau_magnitude, tawss))
        osi[tawss == 0] = 0.0
        values = {
            "TAWSS": tawss,
            "OSI": osi,
            "RRT": divide(np.ones_like(mean_tau_magnitude), mean_tau_magnitude),
            "ECAP": divide(osi, tawss),
            "TWSSG": self._int_twssg / T,
        }
        indices = {}
        for name, value in values.items():
            indices[name] = dolfinx.fem.Function(self._Q, name=name)
            indices[name].x.array[:] = value
        return indices


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


def test_wall_shear_stress_independent_of_pressure():
    """The wall shear stress of the full Cauchy stress equals that of ``Stress``, which omits p."""
    mu = 0.035
    for cell_type in _CELL_TYPES:
        domain, submesh, entity_map = _unit_cube_boundary(cell_type)
        u = dolfinx.fem.Function(dolfinx.fem.functionspace(domain, ("Lagrange", 2, (3,))))
        u.interpolate(lambda x: (x[1] ** 2 - x[2], x[0] * x[2], np.sin(x[0] + x[1])))
        p = dolfinx.fem.Function(dolfinx.fem.functionspace(domain, ("Lagrange", 1)))
        p.interpolate(lambda x: 10 + 50 * x[0] - 20 * x[1] * x[2])
        n = ufl.FacetNormal(domain)

        V_sub = dolfinx.fem.functionspace(submesh, ("DG", 1, (3,)))
        tau_full = dolfinx.fem.Function(V_sub)
        InterpolateDG(wall_shear_stress(cauchy_stress(u, p, mu), n), V_sub, entity_map)(
            tau_full
        )
        tau = Stress(u, V_sub, mu, entity_map)()
        assert np.abs(tau.x.array).max() > 1e-2
        np.testing.assert_allclose(tau.x.array, tau_full.x.array, atol=1e-12)


def test_hemodynamic_indices():
    """Samples ``tau_k = c_k v`` with a fixed vector ``v`` per dof give closed form indices."""
    from mpi4py import MPI

    c = np.array([1.0, 0.5, -0.25, 2.0, -1.5])
    t = np.array([0.0, 0.1, 0.3, 0.4, 0.45])

    _, submesh, _ = _unit_cube_boundary(dolfinx.mesh.CellType.tetrahedron)
    V_sub = dolfinx.fem.functionspace(submesh, ("DG", 1, (3,)))
    rng = np.random.default_rng(MPI.COMM_WORLD.rank)
    index_map = V_sub.dofmap.index_map
    num_dofs = index_map.size_local + index_map.num_ghosts
    v = rng.uniform(-1, 1, size=(num_dofs, 3)) + np.array([2, 0, 0])
    v_magnitude = np.linalg.norm(v, axis=1)

    indices = HemodynamicIndices(V_sub)
    tau = dolfinx.fem.Function(V_sub)
    for c_k, t_k in zip(c, t):
        tau.x.array[:] = (c_k * v).flatten()
        indices.add(tau, t_k)
    assert indices.num_samples == len(c)
    computed = indices.compute()

    def time_average(f):
        """Trapezoidal time average of samples ``f`` at the times ``t``."""
        return 0.5 * np.sum((f[1:] + f[:-1]) * np.diff(t)) / (t[-1] - t[0])

    mean_c = time_average(c)
    mean_c_magnitude = time_average(np.abs(c))
    osi = 0.5 * (1 - abs(mean_c) / mean_c_magnitude)
    expected = {
        "TAWSS": mean_c_magnitude * v_magnitude,
        "OSI": np.full_like(v_magnitude, osi),
        "RRT": 1 / (abs(mean_c) * v_magnitude),
        "ECAP": osi / (mean_c_magnitude * v_magnitude),
        "TWSSG": np.sum(np.abs(np.diff(c))) / (t[-1] - t[0]) * v_magnitude,
    }
    assert computed.keys() == expected.keys()
    for name, value in expected.items():
        assert computed[name].name == name
        np.testing.assert_allclose(computed[name].x.array, value, rtol=1e-12)
    assert 0 <= computed["OSI"].x.array.min() and computed["OSI"].x.array.max() <= 0.5


def test_hemodynamic_indices_zero_wall_shear_stress():
    """Indices are zero instead of dividing by zero, where there is no wall shear stress at all
    (first dofs) and where it oscillates with zero time average (remaining dofs)."""
    c = np.array([1.0, -1.0, 1.0, -1.0])
    t = np.array([0.0, 0.5, 1.5, 2.0])  # Intervals symmetric about t = 1, so <c> = 0

    _, submesh, _ = _unit_cube_boundary(dolfinx.mesh.CellType.tetrahedron)
    V_sub = dolfinx.fem.functionspace(submesh, ("DG", 1, (3,)))
    index_map = V_sub.dofmap.index_map
    num_dofs = index_map.size_local + index_map.num_ghosts
    v = np.tile([0.0, 2.0, 0.0], (num_dofs, 1))
    num_zero = num_dofs // 3
    v[:num_zero] = 0.0

    indices = HemodynamicIndices(V_sub)
    tau = dolfinx.fem.Function(V_sub)
    for c_k, t_k in zip(c, t):
        tau.x.array[:] = (c_k * v).flatten()
        indices.add(tau, t_k)
    with np.errstate(all="raise"):
        computed = indices.compute()

    tawss = 2.0  # <|c|> = 1 times |v| = 2
    expected_nonzero = {"TAWSS": tawss, "OSI": 0.5, "RRT": 0.0, "ECAP": 0.5 / tawss}
    for name, value in expected_nonzero.items():
        np.testing.assert_allclose(computed[name].x.array[:num_zero], 0.0, atol=1e-14)
        np.testing.assert_allclose(computed[name].x.array[num_zero:], value, rtol=1e-12)
    np.testing.assert_allclose(computed["TWSSG"].x.array[:num_zero], 0.0, atol=1e-14)


if __name__ == "__main__":
    test_interpolate_dg_reproduces_polynomial()
    test_stress()
    test_wall_shear_stress_independent_of_pressure()
    test_hemodynamic_indices()
    test_hemodynamic_indices_zero_wall_shear_stress()
