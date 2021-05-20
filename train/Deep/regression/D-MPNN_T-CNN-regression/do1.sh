echo "fold,max(Pearson Correlation),Concordance Index" >> fold_final_.txt
for i in `seq 0 9`
do
  cd rresult_${i}
  grep epoch valid_markdowntable.txt|grep -v "#" > temp
  auc_max=`awk -F "|" '{print $4}' temp |sort -g|tail -1`
  grep ${auc_max} temp > temp1
  num=`wc -l temp1|awk '{print $1}'`
  if [ `echo "$num > 1.0" |bc` -eq 1 ];then aupr=`awk -F "|" '{print $6}' temp1 |sort -g|tail -1`;else aupr=`awk -F "|" '{print $6}' temp1`;fi
  echo -e "fold_${i},\c" >> ../fold_final_.txt
  echo -e "${auc_max},\c" >> ../fold_final_.txt
  echo "${aupr}" >> ../fold_final_.txt
  cd ..
done
