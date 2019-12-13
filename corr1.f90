
! ********************CAOM************************
! !> @autor: Carlos A. Ortiz-Mahecha
!    caraortizmah@gmail.com
!    caraortizmah@unal.edu.co

 
subroutine corr_nsga2(indv, LABEL, TRAIN_CHAR, HASHA, ARRAY_RR, EXP_SCORES, index_n, corr, auc,  count_size)!corr, auc)
  !>
  !Initialization of matrices that depends on the sliding_window function
  !
  implicit none
  
  integer, intent(in) :: count_size
  real, intent(out) :: corr
  !real corr
  real, intent(out) :: auc
  integer, intent(in) :: index_n

  integer i, aux_n, resol
  integer j, k, tmp_index
  real tmp, mean_x, mean_y
  real cov, var_x, var_y
  real min, max, diff_max, diff_min
  real tp, fn, tn, fp, diff_1, diff_2
  real threshold_exp, threshold_pred
  character*20, intent(in) :: LABEL
  real, dimension(9,20) :: ARRAY_RR
  intent(in) ARRAY_RR
  character, dimension(index_n, 9) :: TRAIN_CHAR
  intent(in) TRAIN_CHAR
  character*50, dimension(index_n) :: HASHA
  intent(in) HASHA
  real*4 indv(9)
  intent(in) indv
  real, dimension(count_size) :: EXP_SCORES
  intent(in) EXP_SCORES
  real, dimension(count_size) :: PRED_SCORES
  real, dimension(count_size) :: TPR, FPR
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
  mean_x = SUM(EXP_SCORES) / count_size
  mean_y = SUM(PRED_SCORES) / count_size
  cov = SUM((EXP_SCORES(1:count_size) - mean_x) * (PRED_SCORES(1:count_size) - mean_y))
  var_x = SUM((EXP_SCORES(1:count_size) - mean_x) * (EXP_SCORES(1:count_size) - mean_x))
  var_y = SUM((PRED_SCORES(1:count_size) - mean_y) * (PRED_SCORES(1:count_size) - mean_y))
  corr = (cov / SQRT(var_x)) / SQRT(var_y)
  
  !*******************ROC******************
  max = 0
  DO i = 1, count_size
    IF (max<PRED_SCORES(i)) max = PRED_SCORES(i)
  END DO
  !print *, max
  min = max
  DO i = 1, count_size
    IF (min>PRED_SCORES(i)) min = PRED_SCORES(i)
  END DO
  !print *, min
  
  threshold_exp = 0.426
  !threshold_pred = 0.0
  resol = 40

  diff_max = max - min
  diff_min = diff_max / resol
  threshold_pred = min 
  !print *, max, min
  !print *, diff_max, diff_min

  DO i = 1, resol
    
    threshold_pred = min + (diff_min * i)
    tp = 0 !true positives
    fn = 0 !false negatives
    tn = 0 !true negatives
    fp = 0 !false positives
    DO j = 1, count_size
      IF (EXP_SCORES(j)>threshold_exp) THEN !below of the experimental threshold is negative
        IF (PRED_SCORES(j)>threshold_pred) THEN ! negative for predicted
          tn = tn + 1
        ELSE ! positive for predicted
          fp = fp + 1
        ENDIF
      ELSE !above or equal of the experimental threshold is positive
        IF (PRED_SCORES(j)>threshold_pred) THEN ! negative for predicted
          fn = fn + 1
        ELSE ! positive for predicted
          tp = tp + 1
        ENDIF
      ENDIF
    END DO
    TPR(i) = tp / (tp + fn)
    FPR(i) = fp / (tn + fp)
    !print *, tp, fn, fp, tn, TPR(i), FPR(i), threshold_pred
    !print *, TPR(i), FPR(i)
    !print *, TPR(i), FPR(i), threshold_pred
  END DO
  
  auc = 0
  diff_1 = 0
  diff_2 = 0
  DO i = 1, resol
    !auc = (FPR(i) - diff_1)*TPR(i) + auc !rectangle area
    auc = (FPR(i) - diff_1)*((TPR(i) + diff_2) / 2) + auc !trapezoid area
    diff_1 = FPR(i)
    diff_2 = TPR(i)
  END DO
  !WRITE (*,*) "AUC = ", auc  

  !WRITE (*,*) corr

end subroutine corr_nsga2