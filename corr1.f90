
! ********************CAOM************************
! !> @autor: Carlos A. Ortiz-Mahecha
!    caraortizmah@gmail.com
!    caraortizmah@unal.edu.co

 
subroutine corr_nsga2(indv, LABEL, TRAIN_CHAR, HASHA, ARRAY_RR, EXP_SCORES, index_n, corr)
  !>
  !Initialization of matrices that depends on the sliding_window function
  !
  implicit none
  
  real, intent(out) :: corr
  integer, intent(in) :: index_n

  integer i, aux_n
  integer j, k, tmp_index
  real tmp, mean_x, mean_y
  real cov, var_x, var_y
  character*20, intent(in) :: LABEL
  real, dimension(9,20) :: ARRAY_RR
  intent(in) ARRAY_RR
  character, dimension(index_n, 9) :: TRAIN_CHAR
  intent(in) TRAIN_CHAR
  character*50, dimension(index_n) :: HASHA
  intent(in) HASHA
  real*4 indv(9)
  intent(in) indv
  real, dimension(8424) :: EXP_SCORES
  intent(in) EXP_SCORES
  real, dimension(8424) :: PRED_SCORES
  real, dimension(index_n) :: PR_SC, PR_SC_N
  real, dimension(index_n,9) :: NUM_SCORES

  !********* random assigning of individuals*********

  !b_low  = -1
  !b_up = 1
  !call random_seed()
  !DO i = 1, 10
  !  call random_number(indv(i))
  !  indv(i) = indv(i)*(b_up-b_low) + b_low 
  !END DO
  !WRITE (*, "( 5 f11.6 )" ) indv

  !********* assigning scores*********
  i = 0
  j = 0
  k = 0
  
  DO i = 1, index_n
    DO j = 1, 9
      DO k = 1, 20
        IF (TRAIN_CHAR(i,j) == LABEL(k:k)) THEN

          NUM_SCORES(i,j) = ARRAY_RR(j,k)
        ENDIF
      END DO
    END DO
  END DO
  !WRITE (*,*) NUM_SCORES

  !*********sum of weights*********
  i = 0
  DO i = 1, index_n
  !  WRITE (*,*) NUM_SCORES(i,:)
    PR_SC(i) = SUM(NUM_SCORES(i,:)*indv(:))
  END DO

  ! *********stacking to original size*********
  PR_SC_N = PR_SC

  i = 1
  j = 1
  k = 0
  aux_n = 1
  tmp_index = 1

  DO WHILE (i < index_n)
    DO WHILE (HASHA(i) == HASHA(i+1)) !TR_SC_HA_N
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
    DO WHILE (HASHA(i) /= HASHA(i+1)) !TR_SC_HA_N
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

  ! *********correlating*********
  corr = 0.0
  mean_x = SUM(EXP_SCORES) / 8424
  mean_y = SUM(PRED_SCORES) / 8424
  cov = SUM((EXP_SCORES(1:8424) - mean_x) * (PRED_SCORES(1:8424) - mean_y))
  var_x = SUM((EXP_SCORES(1:8424) - mean_x) * (EXP_SCORES(1:8424) - mean_x))
  var_y = SUM((PRED_SCORES(1:8424) - mean_y) * (PRED_SCORES(1:8424) - mean_y))
  corr = (cov / SQRT(var_x)) / SQRT(var_y)

  !WRITE (*,*) corr

end subroutine corr_nsga2