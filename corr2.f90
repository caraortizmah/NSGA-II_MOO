
! ********************CAOM************************
! !> @autor: Carlos A. Ortiz-Mahecha
!    caraortizmah@gmail.com
!    caraortizmah@unal.edu.co

subroutine store_nsga2(ARRAY_RR, LABEL, index_n, count_size, train_txt)
  !>
  !Initialization of matrices that depends on the sliding_window function
  !
  implicit none

  integer, intent(in) :: count_size
  integer, intent(out) :: index_n

  integer eastat, i, indexn, aux_n
  integer j, k, aux_nn
  character(len=50) aux
  character(len=50), intent(in) :: train_txt
  character(len=50), dimension(20) :: A
  character(len=50), dimension(2)  :: B
  character(len=50), dimension(10,20) :: ARRAY_R
  real, intent(out), dimension(9,20) :: ARRAY_RR
  character(len=50), dimension(count_size,2) :: TRAIN_SCORES!8424
  character(len=50), dimension(count_size,3) :: TR_SC_HA!8424
  character(len=5),  dimension(20) :: PEP
  character(len=50), allocatable :: TR_SC_HA_N(:,:)
  character(len=50), allocatable :: TRAIN_CHAR(:,:)
  character(len=50), dimension(10) :: aux_xx
  character(len=20), intent(out) :: LABEL
  real, dimension(count_size) :: EXP_SCORES!8424
  integer, dimension(20,9) :: AR
  integer, dimension(20,2) :: AA
  integer, dimension(2) :: aux_x
  real, allocatable :: PR_SC(:), PR_SC_N(:)
  real, allocatable :: NUM_SCORES(:,:)

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

  !****matrix training data ****'train1_DRB1_0101_gap.txt'
  OPEN(11, file=train_txt, status='old', action='read', position='rewind')

  DO i = 1, count_size!8424
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
  END DO
  indexn = 0 
  !WRITE (*,*) indexn

  DO i = 1, count_size!  for i in tr_sc_ha:!8424
    aux   = TR_SC_HA(i,1)
    aux_n = len(trim(aux))
    IF (aux_n > 9 ) THEN
      indexn = indexn + 1 + aux_n - 9
    ELSE
      indexn = indexn + 1
    ENDIF
  END DO
  ALLOCATE(TR_SC_HA_N(indexn,3))

  index_n = 0
  DO i = 1, count_size!  for i in tr_sc_ha:8424
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

      END DO
    ELSE
      index_n = index_n + 1
      TR_SC_HA_N(index_n,:) = TR_SC_HA(i,:)
    ENDIF
  END DO

  ALLOCATE(TRAIN_CHAR(indexn,9))
  ALLOCATE(PR_SC(indexn))
  ALLOCATE(PR_SC_N(indexn))
  ALLOCATE(NUM_SCORES(indexn,9))
  
  !**** secondary matrix assignement ****
  DO i = 1, count_size !number of train_scores' rows of the original set 8424
     read(TRAIN_SCORES(i,2),*)EXP_SCORES(i)   !extracting scores of the train_scores vector
     !WRITE (*,*) EXP_SCORES(i)
  END DO

  i = 0
  DO i = 1, index_n
    DO j = 1, 9
      TRAIN_CHAR(i,j) = TR_SC_HA_N(i,1)(j:j)
    END DO
    !WRITE (*,*) TR_SC_HA_N(i,1)
    !WRITE (*,*) TRAIN_CHAR(i,:), '***'
  END DO


  !********* Reorganizing labels*********
  PEP=(/'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y' /)
  AR(:,:)=0
  AA(:,1)=(/1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 /)

  DO i = 1, index_n
    !l = 0
    DO j = 1, 9
      DO k = 1, 20
        aux = TR_SC_HA_N(i,1)(j:j)
        IF (aux == PEP(k)) THEN
          AR(k,j) = AR(k,j) + 1
        ENDIF
      END DO
    END DO
  END DO

  DO i = 1, 20
    AA(i,2) = SUM(AR(i,1:9))
  END DO

  !DO i = 1, 19
  i = 1
  DO WHILE (i < 20)
    IF (AA(i,2) < AA(i+1,2)) THEN
      aux_x(:) = AA(i+1,:)
      AA(i+1,:) = AA(i,:)
      AA(i,:) = aux_x(:)
      IF (i > 1) THEN
        i = i - 1
      ENDIF
    ELSE
      i = i + 1
    END IF
  END DO

  !WRITE (*,*) AA(:,:)

  DO i = 1, 20
    DO j = 1, 20
      aux_n = AA(i,1)
      !aux = ARRAY_R(1,1:20)
      !WRITE (*,*) ARRAY_R(1,j:j)
      IF (PEP(aux_n) == ARRAY_R(1,j)) THEN
        aux_xx(:) = ARRAY_R(:,j)
        ARRAY_R(:,j) = ARRAY_R(:,i)
        ARRAY_R(:,i) = aux_xx(:)
      ENDIF
    END DO
  END DO

  DO j = 1, 20
    LABEL(j:j) = ARRAY_R(1,j)
  END DO
  READ(ARRAY_R(2:10,:),*)ARRAY_RR(:,:)

end subroutine store_nsga2