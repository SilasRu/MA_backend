# transcript_analyser

Follow these steps to setup the project using docker: 

- > docker build --tag transcript_analysis .

- > docker run --mount 'type=volume,src=transcript_analyser_storage,dst=/transcript_analyser_storage' --restart unless-stopped -d -p 47122:47122 --name transcript-analyser transcript_analysis


If you want to run the package locally without docker: 

- > pip install -r requirements.txt

- > JOBS_DIR=jobs INDICES_DIR=indices python -m uvicorn api:app --reload



Samples for all the endpoint with their tests for all different combinations of query parameters are in the following link as a Postman collection:

https://www.getpostman.com/collections/dac30c59045c27a4d8f3
