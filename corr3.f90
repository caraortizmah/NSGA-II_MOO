
! ********************CAOM************************
! !> @autor: Carlos A. Ortiz-Mahecha
!    caraortizmah@gmail.com
!    caraortizmah@unal.edu.co

subroutine store_nsga2(count_size, index_nn, train_txt, TRAIN_C, HASHA, EXP_SCORES)
  !>
  !Initialization of matrices that depends on the sliding_window function
  !
  implicit none

  integer, intent(in) :: count_size
  integer, intent(in) :: index_nn

  integer eastat, i, aux_n, index_n
  integer j, aux_nn
  character(len=50) aux
  character(len=50), intent(in) :: train_txt
  character(len=50), dimension(20) :: A
  character(len=50), dimension(2)  :: B
  character(len=50), dimension(10,20) :: ARRAY_R
  character(len=50), dimension(count_size,2) :: TRAIN_SCORES
  character(len=50), dimension(count_size,3) :: TR_SC_HA
  character(len=50), dimension(index_nn,3) :: TR_SC_HA_N
  character*50, dimension(index_nn) :: HASHA
  intent(out) HASHA
  character, dimension(index_nn,9) :: TRAIN_C
  intent(out) TRAIN_C
  real, intent(out), dimension(count_size) :: EXP_SCORES

  !print *, 'Initialization of the matrices'

  OPEN(15, file='matrix_DR1_label.txt', status='old', action='read', position='rewind')

  DO i = 1, 10
    READ(15,*,iostat=eastat) A
    IF (eastat < 0) THEN
      EXIT
    ELSE IF (eastat > 0) THEN
      STOP 'IO-error'
    ENDIF
    ARRAY_R(i,:)=A

  END DO

  !****matrix training data ****'train1_DRB1_0101_e.txt'
  OPEN(11, file=train_txt, status='old', action='read', position='rewind')

  DO i = 1, count_size
    READ(11,*,iostat=eastat) B
    IF (eastat < 0) THEN
      EXIT
    ELSE IF (eastat > 0) THEN
      STOP 'IO-error'
    ENDIF
    TRAIN_SCORES(i,:)=B
  END DO

  !*******Adding hashable objects in the peptide data*******

  TR_SC_HA(:,1) = TRAIN_SCORES(:,1)
  TR_SC_HA(:,2) = TRAIN_SCORES(:,2)
  TR_SC_HA(:,3) = TRAIN_SCORES(:,1)

  DO i = 1, 3
  aux=TR_SC_HA(i,1)
  !WRITE(*,*) TR_SC_HA(i,1)!(i:i)!aux(i:i)
  END DO

  index_n = 0
  DO i = 1, count_size!  for i in tr_sc_ha:
    aux   = TR_SC_HA(i,1)
    aux_n = len(trim(aux))
    IF (aux_n > 9 ) THEN
      aux_nn = 1 + aux_n - 9
      DO j = 1, aux_nn
        index_n = index_n + 1
        !TR_SC_HA_N(index_n,:) = (/TR_SC_HA(i,1)(j:j+8), TR_SC_HA(i,2), TR_SC_HA(i,3) /)
        TR_SC_HA_N(index_n,1) = TR_SC_HA(i,1)(j:j+8)
        TR_SC_HA_N(index_n,2) = TR_SC_HA(i,2)
        TR_SC_HA_N(index_n,3) = TR_SC_HA(i,3)

        !WRITE(*,*) TR_SC_HA_N(index_n,:)
      END DO
    ELSE
      index_n = index_n + 1
      TR_SC_HA_N(index_n,:) = TR_SC_HA(i,:)
    ENDIF
  END DO
  
  DO i = 1, index_n
    DO j = 1, 9
      TRAIN_C(i,j) = TR_SC_HA_N(i,1)(j:j)
    END DO
  END DO

  HASHA(:) = TR_SC_HA_N(:,3)

  READ(TRAIN_SCORES(:,2),*)EXP_SCORES(:)

end subroutine store_nsga2