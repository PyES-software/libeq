! EQ FORTRAN LIB FOR EQUILIBRIA RESOLUTION
! by Salvador Blasco

module mcpluss
  implicit none (type, external)
  public
contains
subroutine cpluss(C, B, P, NCP, NEQ)
  ! Compute the free concentrations for all the components.
  integer,          intent(in) :: NEQ                        ! number of equilibria
  integer,          intent(in) :: NCP                        ! number of components
  double precision, intent(inout), dimension(NCP+NEQ)  :: C  ! free concentrations
  integer,          intent(in),    dimension(NEQ, NCP) :: P  ! stoichiometry
  double precision, intent(in),    dimension(NEQ)      :: B  ! equilibrium constants

  integer :: i

  do i=1,NEQ
    C(i+NCP) = B(i)*PRODUCT(C(1:NCP)**P(i,:))
  end do
end subroutine cpluss
end module mcpluss
