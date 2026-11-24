! EQ FORTRAN LIB FOR EQUILIBRIA RESOLUTION
! by Salvador Blasco <salvador.blasco@uv.es>

module mouter
    implicit none (type, external)
    public
contains

subroutine outer(correction, C, charge, Bref, Iref, Iback, zstar, DHA, DHB, DHX, T, P, CTRL, NCP, NEQ)
    ! Outer point algorithm for fitting the activities, as described by Carrayrou (2021), 10.1002/aic.17391
    ! For the activity correction an extended Debye-Huckel model is assumed as reported in
    ! Castellino (2023) 10.1016/j.chemolab.2023.104860
    use mnr, only: nr
    integer, intent(in)          :: NCP     ! number of components
    integer, intent(in)          :: NEQ     ! number of equilibria
    integer, intent(inout)       :: CTRL    ! control value. See return values in header of this file.
    double precision, intent(in) :: Iref    ! reference ionic strength
    double precision, intent(in) :: Iback   ! background ionic strength
    double precision, intent(inout), dimension(NEQ)      :: correction  ! the activity correction
    double precision, intent(inout), dimension(NEQ+NCP)  :: C           ! free concentration array
    integer,          INTENT(IN),    DIMENSION(NCP+NEQ)  :: charge      ! the charge of every species
    double precision, intent(in),    dimension(NEQ)      :: Bref        ! the equilibrium constants at Iref
    integer,          intent(in),    dimension(NEQ)      :: zstar       ! z*
    double precision, intent(in)                         :: DHA, DHB    ! debye-huckel parameters A and B
    double precision, intent(in),    dimension(NEQ,3)    :: DHX         ! per-species Debye-Huckel parameters C, D and E
    integer,          intent(in),    dimension(NEQ, NCP) :: P           ! The stoichiometry array
    double precision, intent(in),    dimension(NCP)      :: T           ! analytical concentration values

    DOUBLE PRECISION :: Icalc ! current ionic strength
    DOUBLE PRECISION, dimension(NEQ) :: previous_logcorrection
    DOUBLE PRECISION, dimension(NEQ) :: current_logcorrection
    double precision, parameter :: CORRECTION_TOLERANCE = 1e-5
    double precision, dimension(NEQ)   :: Bcorr        ! corrected beta

    integer :: its
    integer, parameter :: MAXITS = 100

    !previous_logcorrection = LOG10(correction)

    do its=1,MAXITS
      ! I. compute ionic strength
      Icalc = calc_conc_ionic_strength(C, charge, NCP+NEQ) + Iback

      ! II. compute activity coefficients
      call calc_activity_correction(current_logcorrection, Icalc, Iref, zstar, DHA, DHB, DHX(:,1), DHX(:,2), DHX(:,3), NEQ)
      !print "(A, I5, A, 7E12.4)", "outer [", its, "]; correction = ", correction

      ! III. Solve NR
      Bcorr = Bref * 10**current_logcorrection
      call nr(C, T, Bcorr, P, ctrl, NCP, NEQ)
      if (CTRL > 50) return   ! non-recoverable error happened

      ! IV. test convergence
      if (all(abs(current_logcorrection-previous_logcorrection) < CORRECTION_TOLERANCE)) then
          correction = 10**current_logcorrection
          return
      else
          previous_logcorrection = current_logcorrection
      end if
    end do
    CTRL = 4  ! outer point did not converge or lacked iterations to converge
end subroutine outer


subroutine outer_solid(correction, C, CS, charge, Bref, KS, Q, Iref, Iback, zstar, DHA, DHB, DHX, T, P, CTRL, NCP, NEQ, NSL)
    ! Outer point algorithm for fitting the activities, as described by Carrayrou (2021), 10.1002/aic.17391
    ! This routine is for when solids are present.
    ! For the activity correction an extended Debye-Huckel model is assumed as reported in
    ! Castellino (2023) 10.1016/j.chemolab.2023.104860
    use mnr, only: nr_solids
    use solids, only: saturation_product
    integer,          intent(in)          :: NCP     ! number of components
    integer,          intent(in)          :: NEQ     ! number of equilibria
    integer,          intent(in)          :: NSL     ! number of solids
    integer,          intent(inout)       :: CTRL    ! control value. See return values in header of this file.
    double precision, intent(in)          :: Iref    ! reference ionic strength
    double precision, intent(in)          :: Iback   ! background ionic strength
    double precision, intent(inout), dimension(NEQ)      :: correction ! the activity correction
    double precision, intent(inout), dimension(NEQ+NCP)  :: C          ! free concentration array
    double precision, intent(inout), dimension(NSL)      :: CS         ! solid conc initial estimates (in) and refined values (out)
    integer,          INTENT(IN),    DIMENSION(NCP+NEQ)  :: charge     ! the charge of every species
    double precision, intent(in),    dimension(NEQ)      :: Bref       ! the equilibrium constants at Iref
    double precision, intent(in),    dimension(NSL)      :: KS         ! the solubility constant
    integer,          intent(in),    dimension(NEQ)      :: zstar      ! z*
    double precision, intent(in)                         :: DHA, DHB   ! debye-huckel parameters A and B
    double precision, intent(in),    dimension(NEQ,3)    :: DHX        ! per-species Debye-Huckel parameters C, D and E
    integer,          intent(in),    dimension(NEQ, NCP) :: P          ! The stoichiometry array
    integer,          intent(in),    dimension(NSL, NCP) :: Q          ! The solids stoichiometry array
    double precision, intent(in),    dimension(NCP)      :: T          ! analytical concentration values

    DOUBLE PRECISION :: Icalc ! current ionic strength
    DOUBLE PRECISION, dimension(NEQ) :: previous_correction
    DOUBLE PRECISION, dimension(NEQ) :: SI
    double precision, parameter :: CORRECTION_TOLERANCE = 1e-5
    DOUBLE PRECISION, PARAMETER :: si_tolerance = 1e-6
    logical, dimension(NSL) :: do_solid

    integer :: its, i, NX
    integer, parameter :: MAXITS = 100

    previous_correction = correction

    do its=1,MAXITS
      ! I. compute ionic strength
      Icalc = calc_conc_ionic_strength(C, charge, NCP+NEQ) + Iback

      ! II. compute activity coefficients
      call calc_activity_correction(correction, Icalc, Iref, zstar, DHA, DHB, DHX(:,1), DHX(:,2), DHX(:,3), NEQ)
      !print "(A, I5, A, 7E12.4)", "outer [", its, "]; correction = ", correction

      ! III. calculate possible solids
      call saturation_product(C, Q, NCP, NSL, si)
      SI = SI/KS
      do i=1,NSL
        if (si(i) < 1.0 + si_tolerance) then
            if (.not. do_solid(i)) then
                CS(i) = 0.0     ! if there is no solid, make CS=0.0
              end if
          do_solid(i) = .false.  ! remove from model
          cycle
        end if
        if (si(i) > 1.0 + si_tolerance) then
          do_solid(i) = .true.
          if (CS(i) == 0.0) then
              CS(i) = 1e-8
          end if
          cycle
        end if
      end do
      NX = NCP + count(do_solid)
      call nr_solids(C, CS, T, Bref, P, KS, Q, ctrl, do_solid, NCP, NEQ, NSL, NX)

      ! IV. Solve NR
      call nr_solids(C, CS, T, Bref, P, KS, Q, ctrl, do_solid, NCP, NEQ, NSL, NX)
      if (CTRL > 50) return   ! non-recoverable error happened

      ! IV. test convergence
      if (all(abs(correction-previous_correction) < CORRECTION_TOLERANCE)) then
          return
      else
          previous_correction = correction
      end if
    end do
    CTRL = 4  ! outer point did not converge or lacked iterations to converge
end subroutine outer_solid


subroutine calc_activity_correction(corr, I, Iref, zstar, DHA, DHB, DHC, DHD, DHE, NEQ)
    double precision, INTENT(OUT), dimension(NEQ) :: corr  ! the correction for log(beta) from I to Iref
    double precision, intent(in) :: I                      ! the ionic strength to calculate the correction from
    double precision, intent(in) :: Iref                   ! reference ionic strength
    double precision, intent(in) :: DHA, DHB               ! debye-huckel parameters A and B
    double precision, intent(in), dimension(NEQ) :: DHC, DHD, DHE ! per-species debye-huckel parameters C-E
    integer, intent(in),  dimension(NEQ) :: zstar  ! z*
    integer, intent(in)          :: NEQ                     ! the number of equilibria

    corr = - DHA * zstar * (SQRT(I)/(1+DHB*SQRT(I)) - (SQRT(Iref)/(1+DHB*SQRT(Iref)))) &
           + DHC*(I-Iref) &
           + DHD*(I**1.5 - Iref**1.5) &
           + DHE*(I**2-Iref**2)
end subroutine calc_activity_correction


function calc_conc_ionic_strength(C, charge, NSP) result(I)
    ! Calculate the ionic strength due to the species present in the solution
    integer,          intent(in)                 :: NSP     ! number of species (NCP+NEQ)
    double precision, intent(in), DIMENSION(NSP) :: C       ! the free concentrations in mmol/mL
    integer,          intent(in), DIMENSION(NSP) :: charge ! the charges for each species
    DOUBLE PRECISION :: I                                   ! the calculated ionic strength
    I = 0.5*sum(charge**2 * C)
end function calc_conc_ionic_strength


subroutine full_charge(charges, P, NCP, NEQ)
    integer, intent(in)                    :: NCP     ! number of components
    integer, intent(in)                    :: NEQ     ! number of equilibria
    integer, intent(inout), DIMENSION(NCP+NEQ) :: charges ! the charges for each species
    integer, intent(in),    DIMENSION(NEQ,NCP) :: P   ! stoichiometry array

    integer :: i

    do i=1,NEQ
        charges(NCP+i) = sum(charges(1:NCP) * P(i,:))
    end do
end subroutine full_charge


subroutine prepare_dhp(DHP, zstar, charge, P, psDHP, NCP, NEQ)
    integer,          intent(in)                       :: NCP     ! number of components
    integer,          intent(in)                       :: NEQ     ! number of equilibria
    integer,          INTENT(IN),  DIMENSION(NCP+NEQ)  :: charge  ! the charge of the species
    integer,          intent(in),  dimension(NEQ, NCP) :: P       ! The stoichiometry array
    DOUBLE PRECISION, INTENT(IN),  DIMENSION(NEQ, 6)   :: psDHP   ! per-species DH parameters -> c0, c1, d0, d1, e0, e1
    DOUBLE PRECISION, INTENT(OUT), DIMENSION(NEQ,3)    :: DHP     ! per-species global DH params-> C, D and E
    DOUBLE PRECISION, intent(out), dimension(NEQ) :: zstar        ! z* = sum p_i z_i^2 - z_s^2

    integer, dimension(NEQ) :: pstar
    integer :: i

    do i=1,NEQ
      zstar(i) = -charge(NCP+i)**2 + sum(P(:,i)*charge(1:NCP)**2)
      pstar(i) = -1 + sum(P(:,i))
    end do

    do i=1,3
      DHP(:,i) = psDHP(:,i) * pstar + psDHP(:,i+1) * zstar
    end do
end subroutine prepare_dhp

end module mouter
