for confidenceLevel in 0 50 95 99
do
  python ID3.py -c $confidenceLevel
  python ID3.py -c $confidenceLevel -m	
done
python ID3.py -v
