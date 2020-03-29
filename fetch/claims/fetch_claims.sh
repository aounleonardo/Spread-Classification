folder_name=snopes_$(date +'%m_%d')
scrapy crawl snopes -o "$folder_name/claims.json"

cd  $folder_name
jq 'map(select(.rating == "True"))' claims.json > true.json
jq 'map(select(.rating == "False"))' claims.json > false.json
rm claims.json
cd ..

mv $folder_name ../../data/claims/
