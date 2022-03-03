import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.tokenize import sent_tokenize


def lda_analyzer(text: str, n_components=5):
    sentences = sent_tokenize(text)
    count_vec = CountVectorizer(max_df=.8, min_df=2, stop_words='english')

    doc_term_matrix = count_vec.fit_transform(sentences)

    LDA = LatentDirichletAllocation(n_components=n_components, random_state=41)
    LDA.fit(doc_term_matrix)

    topic_values = LDA.transform(doc_term_matrix)

    sentences_topics = topic_values.argmax(axis=1)

    topics_2_words = {
        i: [count_vec.get_feature_names()[i]
            for i in topic.argsort()[-10:]]
        for i, topic in enumerate(LDA.components_)
    }

    topics_2_sentences = {
        topic: [sentences[i] for i in np.where(sentences_topics == topic)[0]]
        for topic in set(sentences_topics)
    }
    return {
        'topics_2_words': topics_2_words,
        'topics_2_sentences': topics_2_sentences
    }


if __name__ == "__main__":
    pass
    # text = "Uh huh  mary  hi  hello, I'm Susan Thompson Resource manager.  Hi, I'm mary Hanson and I'm applying for one of your kitchen jobs.  Great,  here's a copy of my resume.  Great, have a seat mary.Thank you.  Mary, do you have any experience working in the kitchen?
   #   No,  but I want to learn,  I work hard and  I cook a lot at home.  Okay,  well tell me about yourself.  Well  I love to learn new things.  I'm very  organized  and  I follow directions. Exactly.
   #   That's why my boss at my last  job  made me a trainer  and the company actually gave me a special certificate  for coming to work  on time  every day for a year  and  I'm taking an  english class to  improve my writing skills.  That's.Great.  Why did you leave your last job?
   #   It was  graveyard  and  I need to work  days.  Oh I see.  Well what hours can you.Work  from eight am until five pm.
   #   Okay well do you have any questions for me mary?  Yes.  What  kind of training is needed?  Not a lot.
   #   Most new workers can learn everything the  first day.  Do you have any other questions?  No, I don't think so,
   #  but I've heard a lot of good things about your company and I would really like to work here.  Well I have a few more interviews to do today  but I will call you tomorrow if you get the job.  Okay, you are sure. Nice to meet you. Nice meeting.You too. Thank you so much for your time.  Yes, good luck. Thank you."
   # print(lda_analyzer(text=text).get('topics_2_sentences'))
