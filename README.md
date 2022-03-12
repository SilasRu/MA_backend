# transcript_analysis

Follow these steps to setup the project using docker: 

- > docker build --tag transcript_analysis .

- > docker run -p 5000:5000 transcript_analysis


If you want to run the package locally without docker: 

- > pip install -r requirements.txt

- > python -m uvicorn api:app --reload



Samples for all the endpoint with their tests for all different combinations of query parameters are in the following link as a Postman collection:

https://www.getpostman.com/collections/dac30c59045c27a4d8f3