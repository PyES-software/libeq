! EQ FORTRAN LIB FOR EQUILIBRIA RESOLUTION
! by Salvador Blasco <salvador.blasco@uv.es>

module mnr
    implicit none (type, external)
    public
contains

subroutine lesolve(A, b, x, n, CTRL)
  ! Solve system of lineal equations using LAPACK. This routine requires
  ! the -llapack flag at compilation and, probably also -lblas in some systems.
  integer, intent(in) :: n
  double precision, dimension(n,n), intent(in) :: A   ! The nxn array of coefficients
  double precision, dimension(n),   intent(in) :: b   ! the n array of independent terms
  double precision, dimension(n),   intent(out) :: x  ! the n array where the result is returned
  integer, intent(inout) :: CTRL                      ! control value. 0 on no error.

  integer, dimension(n) :: ipiv   ! pivot indices
  integer :: info, nrhs, lda, ldb

  ! External procedures defined in LAPACK
  external dgesv

  ! Store b in x to prevent it from being overwritten by LAPACK
  x = b

  ! Set up solve by LAPACK routine DGESV
  nrhs = 1 ! number of right hand sides in b
  lda = n  ! leading dimension of a
  ldb = n  ! leading dimension of b

  call dgesv(n, nrhs, A, lda, ipiv, x, ldb, info)

  if (info /= 0) then
      CTRL = 99    ! unable to invert matrix
      return
  end if
end subroutine lesolve


subroutine sysolve(A, b, x, n, CTRL)
    ! Solve lineal equations using LAPACK when A is symmetric. This is always the case when
    ! solids are not present. This routine requires
    ! the -llapack flag at compilation and, probably also -lblas in some systems.
    integer, intent(in) :: n
    double precision, dimension(n,n), intent(in) :: A   ! The nxn array of coefficients
    double precision, dimension(n),   intent(in) :: b   ! the n array of independent terms
    double precision, dimension(n),   intent(out) :: x  ! the n array where the result is returned
    integer, intent(inout) :: CTRL                      ! control value. 0 on no error.

    ! For DGELSY
    integer, dimension(n) :: ipiv   ! pivot indices
    integer :: nrhs, lda, ldb, lwork, info
    double precision, allocatable, dimension(:) :: work
    double precision, dimension(n,n) :: AA

    ! External procedures defined in LAPACK
    external dsysv

    nrhs = 1
    lda = n; ldb = n
    lwork = -1              ! workspace query first
    x = b                   ! b is input and overwritten with the value of x
    AA = A

    lwork = -1
    allocate(work(1))
    call dsysv("U", n, 1, AA, n, ipiv, x, n, work, lwork, info)
    lwork = nint(work(1))
    deallocate(work)

    allocate(work(lwork))
    call dsysv("U", n, 1, AA, n, ipiv, x, n, work, lwork, info)
    deallocate(work)

    if (info /= 0) then
        CTRL = 99    ! unable to invert matrix
        return
    end if
end subroutine sysolve


subroutine jacobian_soluble(J, C, P, NCP, NEQ)
  ! calculate the jacobian df/dlogc of the mass balance not considering solids
  use err, only: amatrix
  double precision, intent(inout), dimension(NCP, NCP) :: J   ! jacobian
  double precision, intent(in),    dimension(NCP+NEQ)  :: C   ! free concentrations
  integer,          intent(in),    dimension(NEQ, NCP) :: P   ! The soluble stoichiometry array
  integer, intent(in) :: NCP                                  ! number of components
  integer, intent(in) :: NEQ                                  ! number of equilibria
  J(:,:) = 0.0
  call amatrix(J, C, P, NCP, NEQ)
end subroutine jacobian_soluble


subroutine jacobian_solid(JC, C, CS, KS, P, Q, NCP, NEQ, NSL, NX, do_solids)
  integer, intent(in) :: NCP                                ! number of components
  integer, intent(in) :: NEQ                                ! number of equilibria
  integer, intent(in) :: NSL                                ! number of solids
  integer, intent(in) ::  NX
  integer,          intent(in),  dimension(NEQ, NCP) :: P  ! The soluble stoichiometry array
  integer,          intent(in),  dimension(NSL, NCP) :: Q  ! The solids stoichiometry array
  double precision, intent(in),  dimension(NEQ+NCP)  :: C  ! free concentation initial estimates (in) and refined values (out)
  double precision, intent(in),  dimension(NSL)      :: KS ! the solubility product
  double precision, intent(in),  dimension(NSL)      :: CS
  double precision, intent(out), dimension(NX, NX)   :: JC  ! jacobian
  logical,          intent(in),  dimension(NSL)      :: do_solids

  integer :: i, j, k
  double precision :: aux
  JC(:,:) = 0.0
  call jacobian_soluble(JC(1:NCP,1:NCP), C, P, NCP, NEQ)
  k = 0
  do i=1,NSL
    if (.not. do_solids(i)) cycle
    k = k + 1
    do j=1,NCP
      JC(j,NCP+k) = Q(k,j)*CS(k)
      aux = -Q(k,j) * product(C(1:NCP)**Q(k,:))
      JC(NCP+k,j) = aux/KS(k)
    end do
  end do
end subroutine jacobian_solid


subroutine nr(C, T, B, P, ctrl, NCP, NEQ)
  ! Perform the actual Newton-Raphson iterations. The initial value is supplied
  ! in the C variable and it is replaced with the refined value upon exit.
  ! The free concentrations are refined in the logarithmic space.
  use mcpluss, only: cpluss
  use mb, only: mb_soluble
  integer, intent(in) :: NCP                               ! number of components
  integer, intent(in) :: NEQ                               ! number of equilibria
  integer, intent(inout) :: ctrl                           ! ctrl -> control value.
  integer, dimension(NEQ, NCP), intent(in) :: P            ! stoichiometry array
  double precision, dimension(NCP+NEQ), intent(inout) :: C ! initial estimates (in) and refined values (out)
  double precision, dimension(NCP), intent(in) :: T        ! analytical concentration values
  double precision, dimension(NEQ), intent(in) ::  B       ! the equilibrium constants

  integer :: its
  double precision, dimension(NCP)      :: X               ! natural log of free concentrations
  double precision, dimension(NCP)      :: fvec
  double precision, dimension(NCP, NCP) :: jac
  double precision, dimension(NCP)      :: dx
  logical :: check, converged
  integer, parameter :: MAXITS=100

  X(:) = LOG(C(1:NCP))

  call cpluss(C, B, P, NCP, NEQ)
  call mb_soluble(fvec, C, P, T, NCP, NEQ)
  dx(:) = 0.0

  call is_root(fvec, NCP, check)  ! check whether initial guess
  if (check) then                 ! is already a root
    CTRL=1
    return
  end if

  ! Main loopC(i) -
  do its=1,MAXITS
    call jacobian_soluble(jac, C, P, NCP, NEQ)

    ! Solve linear system.
    call sysolve(jac, fvec, dx, NCP, CTRL)
    if (CTRL == 99) return   ! matrix could not be inverted
    call limit_step(dx, NCP)

    x = x - dx               ! dx comes sign inverted from lsolve
    C(1:NCP) = EXP(X(1:NCP))

    call cpluss(C, B, P, NCP, NEQ)
    call mb_soluble(fvec, C, P, T, NCP, NEQ)

    ! test for convergence on relative values
    call is_converged(dx, fvec, x, NCP, converged)
    if (converged) then
       ctrl = 0  ! normal exit
       return
    end if
  end do
  ctrl = 2 ! Exit on MAXITS
end subroutine nr


subroutine nr_solids(C, CS, T, B, P, KS, Q, ctrl, do_solids, NCP, NEQ, NSL, NX)
  ! Perform the actual Newton-Raphson iterations. The initial value is supplied
  ! in the X variable and it is replaced with the refined value upon exit.
  use mcpluss, only: cpluss
  use mb, only: mb_solids
  use solids, only: saturation_product
  integer, intent(in)    :: NCP                            ! number of components
  integer, intent(in)    :: NEQ                            ! number of equilibria
  integer, intent(in)    :: NSL                            ! number of solids
  integer, intent(inout)    :: NX                             ! number of refined variables
  integer, intent(inout) :: ctrl                           ! control value
  integer,          intent(in),    dimension(NEQ, NCP) :: P         ! stoichiometry array
  double precision, intent(inout), dimension(NCP+NEQ)  :: C         ! initial estimates (in) and refined values (out)
  double precision, intent(inout), dimension(NSL)      :: CS        ! solid conc initial estimates (in) and refined values (out)
  double precision, intent(in),    dimension(NCP)      :: T         ! analytical concentration values
  double precision, intent(in),    dimension(NEQ)      :: B         ! the equilibrium constants
  integer,          intent(in),    dimension(NSL, NCP) :: Q         ! the equilibrium constants
  double precision, intent(in),    dimension(NSL)      :: KS        ! the equilibrium constants
  logical,          intent(inout), dimension(NSL)      :: do_solids ! which solids are in the model

  integer :: its
  double precision, dimension(NX)       :: X  ! 1:NCP -> concentrations in log space
  double precision, dimension(NCP+NSL)  :: full_fvec
  double precision, dimension(NX)       :: fvec
  double precision, dimension(NX, NX)   :: jac
  double precision, dimension(NX)       :: dx
  double precision, dimension(NSL)      :: SI
  logical :: check, converged
  integer, parameter :: MAXITS=100
  integer :: i, k

  X(1:NCP) = LOG(C(1:NCP))  ! initialise C in log space
  X(NCP+1:NX) = LOG(pack(CS, do_solids)) ! and CS in linear space

  call cpluss(C, B, P, NCP, NEQ)
  call mb_solids(full_fvec(1:NCP), C, P, T, Q, CS, NCP, NEQ, NSL)
  call saturation_product(C, Q, NCP, NSL, SI)
  full_fvec(NCP+1:NCP+NSL) = 1.0 - SI/KS
  call is_root(full_fvec, NCP+NSL, check)  ! check whether initial guess
  if (check) then                 ! is already a root
    CTRL=1
    return
  end if

  dx(:) = 0.0
  fvec(1:NCP) = full_fvec(1:NCP)
  fvec(NCP+1:NX) = pack(full_fvec(NCP+1:NCP+NSL), do_solids)

  ! Main loop
  do its=1,MAXITS
    call jacobian_solid(jac, C, CS, KS, P, Q, NCP, NEQ, NSL, NX, do_solids)

    ! Solve linear system.
    call lesolve(jac, fvec, dx, nx, CTRL)
    if (CTRL == 99) return

    ! if one of the solids concentration wants to go very very low, it may make the jacobian
    ! singular.

    call limit_step(dx, nx)

    x = x - dx
    C(1:NCP) = EXP(X(1:NCP))
    CS = unpack(EXP(X(NCP+1:NX)), do_solids, CS)

    call cpluss(C, B, P, NCP, NEQ)

    ! if one of the solids concentrations goes 0 we cannot continue because the jacobian
    ! would be singular in the next iteration. Best option is to finish iteration and
    ! let the caller function to change the do_solids flags.
    do i=1,NSL
        k = 0
        if (do_solids(i)) then
            k=k+1
            if (CS(i) < 1D-45 .and. SI(i) < 1D0 .and. dx(NCP+k)>10.0) then
                do_solids(i) = .false.
                NX = NX - 1
                CTRL = 6
                return
            end if
            if (CS(i) < 1D-45 .and. SI(i) > 1D0) then
                do_solids(i) = .true.
                NX = NX + 1
                CTRL = 6
                return
            end if
        end if
    end do

    call mb_solids(full_fvec, C, P, T, Q, CS, NCP, NEQ, NSL)
    call saturation_product(C, Q, NCP, NSL, SI)
    full_fvec(NCP+1:NCP+NSL) = 1.0 - SI/KS
    fvec(1:NCP) = full_fvec(1:NCP)
    fvec(NCP+1:NX) = pack(full_fvec(NCP+1:NCP+NSL), do_solids)

    ! test for convergence on relative values
    call is_converged(dx, fvec, x, NX, converged)
    if (converged) then
       ctrl = 0  ! normal exit
       return
    end if
  end do
  ctrl = 2 ! Exit on MAXITS
end subroutine nr_solids


subroutine is_converged(dx, fvec, x, fsize, retval)
    ! test for convergence on relative values
    integer,          intent(in)                   :: fsize
    double precision, intent(in), dimension(fsize) :: dx, fvec, x
    logical,          intent(inout)                :: retval

    double precision, parameter :: REL_TOL  = 1.0d-15   ! relative step
    double precision, parameter :: ABS_TOL  = 1.0d-17   ! absolute step floor
    double precision, parameter :: FRES_TOL = 1.0d-15   ! residual (RMS or max-norm)

    double precision :: max_rel_step, max_abs_res

    integer :: i

    retval = .false.

    ! residual test
    max_abs_res = maxval(abs(fvec))
    if (max_abs_res <= FRES_TOL) then
        retval = .true.
        return
    end if

    ! mixed relative/absolute step test
    max_rel_step = 0.0d0
    do i = 1, fsize
        max_rel_step = max( max_rel_step, &
                            abs(dx(i)) / (abs(x(i)) + ABS_TOL) )
    end do

    if (max_rel_step <= REL_TOL) then
        retval = .true.
    end if
end subroutine is_converged


subroutine scale(M, V, D, n)
    double precision, intent(inout), dimension(n,n) :: M
    double precision, intent(inout), dimension(n) :: V
    double precision, intent(inout), dimension(n) :: D
    integer, intent(in) :: n

    integer :: i, j
    do i=1,n
      D(i) = abs(M(i,i))
    end do
    if (any(D == 0.0)) then   ! if there is any diagonal value equal to zero
        D(:) = 1.0              ! do not scale
        return
    end if
    do i=1,n
      do j=1,n
        M(i, j) = M(i, j) / sqrt(D(i)*D(j))
      end do
      v(i) = v(i) / sqrt(D(i))
    end do
end subroutine scale


subroutine is_root(fvec, NCP, check)
  ! Test for initial guess being a root. Use more stringent test than simply TOLF.
  double precision, intent(in), dimension(NCP) :: fvec
  integer, intent(in) :: NCP     ! number of components
  logical, intent(inout) :: check

  double precision :: test
  integer :: i

  check = .false.
  test=0.0
  do i=1,NCP
    if (abs(fvec(i)) > test) test=abs(fvec(i))
  end do
  if (test < 1D-25) then
    check=.true.
  end if
end subroutine is_root


subroutine update_solids(do_solids, si, NSL, NX)
    ! update the solids that are accounted for in the model according to the procedure outlined
    ! in GEOCHEMICAL AND BIOGEOCHEMICAL REACTION MODELING, 2nd ed. CRAIG M. BETHKE 4.4.3p69
    double precision, intent(in),    dimension(NSL) :: si         ! the solubility index
    logical,          intent(inout), dimension(NSL) :: do_solids  ! the solids to be included
    integer,          intent(in) :: NSL, NX                       ! the number of solids
    double precision, parameter :: SOLUBILITY_TOLERANCE = 1.0E-6

    integer :: idx

    if (any(si < 1.0 + SOLUBILITY_TOLERANCE)) then    ! Any undersaturated minerals in the system?
        idx = minloc(si, dim=1)                       ! remove most undersaturated component
        do_solids(idx) = .false.
        return
    end if

    if (any(si > 1.0 - SOLUBILITY_TOLERANCE)) then    ! Any supersaturated mineral?
        idx = maxloc(si, dim=1)                       ! add the most saturated one to the model
        do_solids(idx) = .true.
        return
    end if
end subroutine update_solids


subroutine limit_step(incrx, n)
    ! Limit the increment vector to a maximum value. If the largest component is smaller
    ! than the max step, nothing is done. If it is, the vector is scaled so that the
    ! largest step is the max step allowed and the rest are scaled accordingly.
    integer, intent(in) :: n   ! the size of the vector
    double precision, intent(inout), dimension(n) :: incrx  ! the vector to be scaled
    double precision, parameter :: MAX_STEP = 20.0

    if (any(abs(incrx) > MAX_STEP)) then
        incrx = incrx * MAX_STEP / MAXVAL(abs(incrx))
    end if
end subroutine limit_step
end module mnr
