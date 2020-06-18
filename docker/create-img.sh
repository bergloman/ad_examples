# cat ../requirements.txt | grep -v "tensorflow" > ./requirements.txt
cp ../requirements.txt ./requirements.txt
sed -i  's/>=/==/g' ./requirements.txt
docker build . -t bergloman:ad_examples
