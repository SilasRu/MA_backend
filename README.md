# transcript_analysis

Follow these steps to setup the project using docker: 

- > sudo docker build --tag transcript_analysis .

- > sudo docker run -p 5000:5000 transcript_analysis


If you want to run the package locally without docker: 

- > pip install -r requirements.txt

- > python -m uvicorn api:app --reload