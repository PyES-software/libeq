! EQ FORTRAN LIB FOR EQUILIBRIA RESOLUTION
! by Salvador Blasco <salvador.blasco@uv.es>
!
! This file contains routines related to solubility

module solids
    implicit none (type, external)
    public
contains

subroutine saturation_product(C, Q, NCP, NSL, SI)
  ! Calculate the saturation product, defined as
  ! SI_i = \prod_j^{NSL} c_j^{q_{ji}}
  integer, intent(in) :: NCP                  ! number of components
  integer, intent(in) :: NSL                  ! number of solids
  double precision, dimension(NCP),      intent(in)  :: C  ! the free concentrations
  integer,          dimension(NSL, NCP), intent(in)  :: Q  ! the solid stoichiometry array
  double precision, dimension(NSL),      intent(out) :: SI ! the solubility product

  double precision :: aux
  integer :: i, j

  SI(:) = 0.0
  do i=1,NSL
    aux = 1.0
    do j=1,NCP
      aux = aux*C(j)**Q(i,j)
    end do
    SI(i) = aux
  end do
end subroutine saturation_product
end module solids
