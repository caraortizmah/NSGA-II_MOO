
!module mine
 ! implicit none

 !PUBLIC :: ARRAY_R

  !contains
Program  test_nsga2_9w
!subroutine corr_nsga2(corr)
  !>
  !Initialization of matrices that depends on the sliding_window function
  !
  implicit none
  
  !real, intent(out) :: corr

  integer eastat, i, indexn, aux_n
  integer j, k, index_n, aux_nn, tmp_index
  real b_low, b_up, tmp, mean_x, mean_y
  real cov, var_x, var_y, corr
  character(len=50) ven, aux
  character(len=50), dimension(20) :: A
  character(len=50), dimension(2)  :: B
  character(len=50), dimension(10,20) :: ARRAY_R
  character(len=50), dimension(8424,2) :: TRAIN_SCORES
  character(len=50), dimension(8424,3) :: TR_SC_HA
  character(len=5),  dimension(20) :: PEP
  character(len=50), allocatable :: TR_SC_HA_N(:,:)
  character(len=50), allocatable :: TRAIN_CHAR(:,:)
  character(len=50), dimension(10) :: aux_xx
  real*4 indv(9)
  real, dimension(8424) :: EXP_SCORES
  real, dimension(8424) :: PRED_SCORES
  integer, dimension(20,9) :: AR
  integer, dimension(20,2) :: AA
  integer, dimension(2) :: aux_x
  real, allocatable :: PR_SC(:), PR_SC_N(:)
  real, allocatable :: NUM_SCORES(:,:)

  print *, 'Initialization of the matrices'

  OPEN(15, file='matrix_DR1_label.txt', status='old', action='read', position='rewind')

  DO i = 1, 10
    READ(15,*,iostat=eastat) A
    IF (eastat < 0) THEN
      EXIT
    ELSE IF (eastat > 0) THEN
      STOP 'IO-error'
    ENDIF
    !WRITE(*,*) ' :inputline =', A
    ARRAY_R(i,:)=A

  END DO

    !****matrix training data ****
  OPEN(11, file='train1_DRB1_0101_e.txt', status='old', action='read', position='rewind')

  DO i = 1, 8424
    READ(11,*,iostat=eastat) B
    IF (eastat < 0) THEN
      EXIT
    ELSE IF (eastat > 0) THEN
      STOP 'IO-error'
    ENDIF
    !WRITE(*,*) ' :inputline =', B
    TRAIN_SCORES(i,:)=B
  END DO

  print *, '*Adding hashable objects in the peptide data'

  TR_SC_HA(:,1) = TRAIN_SCORES(:,1)
  TR_SC_HA(:,2) = TRAIN_SCORES(:,2)
  TR_SC_HA(:,3) = TRAIN_SCORES(:,1)

  !WRITE (*,*) TR_SC_HA(1,:)
  !WRITE (*,*) TR_SC_HA(8424,3)
  !ven = TR_SC_HA(8424,3)
  !WRITE(*,*) ven(1:1+8)
  !WRITE(*,*) ven(6:15)
  !WRITE(*,*) ven(16:20)
  !WRITE(*,*) ven(21:30)
  !WRITE(*,*) len(trim(ven))
  DO i = 1, 3
  aux=TR_SC_HA(i,1)
  !WRITE(*,*) TR_SC_HA(i,1)(i:i)!aux(i:i)
  END DO
  indexn = 0 
  WRITE (*,*) indexn

  DO i = 1, 8424!  for i in tr_sc_ha:
    aux   = TR_SC_HA(i,1)
    aux_n = len(trim(aux))
    IF (aux_n > 9 ) THEN
      indexn = indexn + 1 + aux_n - 9
    ELSE
      indexn = indexn + 1
    ENDIF
  END DO
  ALLOCATE(TR_SC_HA_N(indexn,3))
  !character(len=50), dimension(indexn,3) :: TR_SC_HA_N

  index_n = 0
  DO i = 1, 8424!  for i in tr_sc_ha:
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

        !np.insert(tr_sc_ha_n,index+j+1,np.array((i[0][j:j+9],i[1],i[2])),0)
      !tr_sc_ha_n = np.delete(tr_sc_ha_n, index, axis=0)
        !WRITE(*,*) TR_SC_HA_N(index_n,:)
      END DO
    ELSE
      index_n = index_n + 1
      TR_SC_HA_N(index_n,:) = TR_SC_HA(i,:)
    ENDIF
    !WRITE(*,*) TR_SC_HA_N(index_n,:)
    !WRITE(*,*) TR_SC_HA(i,:)
  END DO

  !WRITE (*,*) TR_SC_HA(8424,3)
  !WRITE (*,*) TR_SC_HA(8424,1)
  !WRITE (*,*) TR_SC_HA_N(8424,3)
  WRITE (*,*) indexn, index_n
  ALLOCATE(TRAIN_CHAR(indexn,9))
  ALLOCATE(PR_SC(indexn))
  ALLOCATE(PR_SC_N(indexn))
  ALLOCATE(NUM_SCORES(indexn,9))
  
  !**** secondary matrix assignement ****
  DO i = 1, 8424 !number of train_scores' rows of the original set
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
  !PRINT *, ARRAY_R

  b_low  = -1
  b_up = 1
  call random_seed()
  DO i = 1, 10
    call random_number(indv(i))
    indv(i) = indv(i)*(b_up-b_low) + b_low 
  END DO
  WRITE (*, "( 5 f11.6 )" ) indv
  !indv(:) = 1.0
  indv(1) = 0.3 
  indv(2) = 0.3 
  indv(3) = 0.3 
  indv(4) = 0.3 
  indv(5) = 0.3 
  indv(6) = 0.3 
  indv(7) = 0.3 
  indv(8) = 0.3 
  indv(9) = 0.3 

  i = 0
  j = 0
  k = 0

  print *, indexn, index_n

  DO i = 1, index_n
    DO j = 1, 9
      DO k = 1, 20
        IF (TRAIN_CHAR(i,j) == ARRAY_R(1,k)) THEN
          read(ARRAY_R(j+1,k),*)NUM_SCORES(i,j)
        ENDIF
      END DO
    END DO
  END DO
  !WRITE (*,*) NUM_SCORES

  i = 0
  DO i = 1, index_n
  !  WRITE (*,*) NUM_SCORES(i,:)
    PR_SC(i) = SUM(NUM_SCORES(i,:)*indv(:))
  END DO

  !print *, 'TR_SC_HA_N: ', TR_SC_HA_N(1:11,3)
  PR_SC_N = PR_SC
  !print *, index_n
  i = 1
  j = 1
  k = 0
  aux_n = 1
  !tmp = PR_SC(1)
  tmp_index = 1
  DO WHILE (i < index_n)
    DO WHILE (TR_SC_HA_N(i,3) == TR_SC_HA_N(i+1,3))
      IF (aux_n  == 1) tmp = PR_SC(i)
      IF (tmp < PR_SC(i+1)) tmp = PR_SC(i+1)
      aux_n = aux_n + 1
      i = i + 1
    END DO
    IF (aux_n > 1) THEN
      PRED_SCORES(j) = tmp
      aux_n = 1
      j = j + 1
      IF (i == index_n) GO TO 100
      tmp = PR_SC(i+1)
      i = i + 1
    ENDIF
    DO WHILE (TR_SC_HA_N(i,3) /= TR_SC_HA_N(i+1,3))
      !IF (aux_n == 1) THEN
      PRED_SCORES(j) = PR_SC(i)
      j = j + 1
      !ENDIF
      i = i + 1
      !tmp = PR_SC(i)
    END DO
    IF (i == index_n) PRED_SCORES(j) = PR_SC(i)
  END DO
  
  100 corr = 0.0 !print *, 'GO TO 100 '

!    IF (TR_SC_HA_N(i,3) /= TR_SC_HA_N(i+1,3)) 
!      k = k + 1
!    ENDIF
!    IF (k == 2) THEN
!      PRED_SCORES(j) = PR_SC(i)
!      k = 0
!      j = j + 1
!    ENDIF
!    i = i + 1
!  END DO
!
!  DO WHILE (i < 40)!index_n)
!    IF (TR_SC_HA_N(i,3) == TR_SC_HA_N(i+1,3)) THEN
!      IF (j == 1) THEN
!        tmp = PR_SC(i) 
!        tmp_index = i
!      ENDIF
!      IF (PR_SC_N(i+1) /= 0.00000000) THEN
!        IF (tmp > PR_SC(i+1)) THEN
!          PR_SC_N(i+1) = PR_SC_N(i+1) * 0.00000000
!          aux_n = aux_n + 1
!        ELSE
!          tmp = PR_SC(i+1)
!          PR_SC_N(tmp_index) = PR_SC_N(i+1) * 0.0000000000
!          tmp_index = i
!          aux_n = aux_n + 1
!        ENDIF
!      ENDIF
!      j = j + 1
!    ELSE
!      j = 1
!      k = k + 1
!
!    ENDIF
!    i = i + 1
!  END DO
!  print *, PR_SC_N(20:40)
!  j = 1
!  aux_n = 1
!  print *, '************'
!  DO i = 20, 40!index_n
!    !IF (j == 8423) STOP
!    print *, i, PR_SC_N(i)
!    IF (PR_SC_N(i) > 0.0) THEN
!      j = j + 1
!      print *, 'true'
!    ELSE
!            print *, 'False'
!      aux_n = aux_n + 1
!    ENDIF
!  END DO
!  !print *, i, j
  !print *, 'j: ', j
  print *, 'aux_n: ', aux_n, 'i ', i, ' j ', j
  !stop
  !print *, PR_SC_N(24) > 0.0
  !WRITE (*,*) size(PRED_SCORES)
  !WRITE (*,*) size(EXP_SCORES)
  WRITE(*,*) "exp: ", EXP_SCORES(8001:8011)
  !WRITE(*,*) "pred PR_SC: ", PR_SC(1:11)
  !WRITE(*,*) "pred PR_SC_N: ", PR_SC_N(1:11)
  WRITE(*,*) "pred: ", PRED_SCORES(8001:8011)
  !stop
  !WRITE(*,*) "hash: ", TR_SC_HA_N(1:14,3)
  
  corr = 0.0
  mean_x = SUM(EXP_SCORES) / 8424
  mean_y = SUM(PRED_SCORES) / 8424
  cov = SUM((EXP_SCORES(1:8424) - mean_x) * (PRED_SCORES(1:8424) - mean_y))
  var_x = SUM((EXP_SCORES(1:8424) - mean_x) * (EXP_SCORES(1:8424) - mean_x))
  var_y = SUM((PRED_SCORES(1:8424) - mean_y) * (PRED_SCORES(1:8424) - mean_y))
  corr = (cov / SQRT(var_x)) / SQRT(var_y)

  WRITE (*,*) corr


!end subroutine label

!end module mine

!program test_nsga2_9w
 !       use mine
  !      implicit none
      !  integer idorg, idnew, ipos
    !    idorg = 56

  !      call label()
        !PRINT *, ARRAY_R
!end subroutine corr_nsga2
end program test_nsga2_9w
