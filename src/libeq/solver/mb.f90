! EQ FORTRAN LIB FOR EQUILIBRIA RESOLUTION
! by Salvador Blasco
!
! This file contains subroutines for mass balance resolution

module mb
    implicit none (type, external)
    public
contains

subroutine mb_soluble(F, C, P, T, NCP, NEQ)
  ! Compute the objective function not considering solids.
  ! f_i = c_i + \sum_{j=1}^E { p_{ji}c_{j+S} } - T_i
  integer, intent(in) :: NEQ  ! number of equlibria
  integer, intent(in) :: NCP  ! number of components
  double precision, dimension(NCP),      intent(out) :: F  ! result output
  double precision, dimension(NCP),      intent(in)  :: T  ! the analytical concentration
  double precision, dimension(NCP+NEQ),  intent(in)  :: C  ! the free concentrations
  integer,          dimension(NEQ, NCP), intent(in)  :: P  ! the soluble stoichiometry

  integer :: i

  DO i=1,NCP
    F(i) = C(i) + sum(P(:,i)*C(NCP+1:NCP+NEQ)) - T(i)
  END DO
END subroutine mb_soluble


subroutine mb_solids(F, C, P, T, Q, CS, NCP, NEQ, NSL)
  ! Compute the objective function when solids are present.
  ! f_i = c_i + \sum_{j=1}^E { p_{ji}c_{j+S} } - T_i + \sum_j^{solids} {q_{ij}S_j}
  integer, intent(in) :: NEQ  ! number of equlibria
  integer, intent(in) :: NCP  ! number of components
  integer, intent(in) :: NSL  ! number of solids
  double precision, dimension(NCP),      intent(out) :: F  ! result output
  double precision, dimension(NCP),      intent(in)  :: T  ! the analytical concentration
  integer,          dimension(NEQ, NCP), intent(in)  :: P  ! the soluble stoichiometry
  integer,          dimension(NSL, NCP), intent(in)  :: Q  ! the solid stoichiometry
  double precision, dimension(NCP+NEQ),  intent(in)  :: C  ! the free concentrations
  double precision, dimension(NSL),      intent(in)  :: CS ! the solids "concentrations"

  integer :: i

  call mb_soluble(F, C, P, T, NCP, NEQ)
  do i=1,NCP
    F(i) = F(i) + sum(CS*Q(:,i))
  end do
END subroutine mb_solids
end module mb
