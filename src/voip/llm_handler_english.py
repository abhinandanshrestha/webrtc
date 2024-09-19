from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
import os

def get_embedding(text):
    url = "http://192.168.88.40:8026/embed"
    payload = {"inputs": text}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    return  response.json()

def generate_response_embeddings():

        positive_responses_1 = ["Yes ", "Yes it's john", "Yes, Yep", "Yes, I am John Doe ", "Yes I'm John Doe", "Yep", "Yeah, I'm John Doe", "Yeah I'm John Doe Speaking","Yeah, tell me"]
        negative_responses_1 = ["No",  "Nope", "No, I'm Suraj", "Nah", "No I'm not John Doe", "I am not John Doe", "Not John Doe","Sorry I'm not John Doe"]
        
        positive_responses_2 = ["Yes","Sure", "Yes Please", "Okay"," Yeah, I have some time.","Yes I have a moment","You can say"]
        negative_responses_2 = ["call me later","not today","not","I'm in middle of something", "Busy right now","busy","No", "No I don't have time", "nope", "not okay", "later", "nop","I don't have a moment","Don't say","no thank you"]
        
        positive_responses_3 = ["Yes","Sure", "Yes Please", "Okay", "Yes more details", "more details","Tell me more", "Yeah i want to know about it more","I'm interested","Interested"]
        negative_responses_3 = ["call me later","not today","busy","I have meeting to attend","I'm in middle of something","No", "No I don\'t have time","No I don\'t want more details","i'll talk later", "nope", "not okay", "later", "nop","no thank you"]
        
        positive_responses_4 = ["Yes","Sure", "Yes Please", "Okay", "Yes as soon as possible"]
        negative_responses_4 = ["call me later","Send me next day","not today", "No", "No I don't have time", "nope", "not okay", "later", "nop","no thank you"]
        
        positive_responses = [positive_responses_1, positive_responses_2, positive_responses_3, positive_responses_4]
        negative_responses = [negative_responses_1, negative_responses_2, negative_responses_3, negative_responses_4]
    
    
        repeat_responses = [["could you speak louder","i can't hear you","repeat","repeat please","can you speak alittle louder?", "pardon?", "can you repeat?", "could you repeat?"]]

        positive_file_path = 'Model/positive_embeddings.npy'
        negative_file_path = 'Model/negative_embeddings.npy'
        repeat_file_path = "Model/repeat_embeddings.npy"
        
        if os.path.exists(repeat_file_path):
            loaded_embeds = np.load(repeat_file_path)
            repeat_embeds = loaded_embeds.tolist()
        else:
            repeat_embeds = [[get_embedding(item) for item in group] for group in repeat_responses]
            # np.save(negative_file_path, np.array(negative_embeds))
        
        if os.path.exists(positive_file_path):
            loaded_embeds = np.load(positive_file_path)
            positive_embeds = loaded_embeds.tolist()
        else:
            positive_embeds = [[get_embedding(item) for item in group] for group in positive_responses]
            # np.save(positive_file_path, np.array(positive_embeds))
            
        if os.path.exists(negative_file_path):
            loaded_embeds = np.load(negative_file_path)
            negative_embeds = loaded_embeds.tolist()
        else:
            negative_embeds = [[get_embedding(item) for item in group] for group in negative_responses]
            # np.save(negative_file_path, np.array(negative_embeds))
            
        return positive_embeds, negative_embeds, repeat_embeds


def top5avg(similarity_list):
    sorted_numbers = sorted(similarity_list, reverse=True)
    top_5 = sorted_numbers[:5]
    return sum(top_5) / len(top_5)

def find_similarity(input_embedding, positive_embeddings, negative_embeddings, repeat_embeddings, threshold=0.5):

    # Calculate cosine similarity between input string and responses

    positive_similarity_scores = [cosine_similarity(input_embedding, embedding)[0][0] for embedding in positive_embeddings]
    negative_similarity_scores = [cosine_similarity(input_embedding, embedding)[0][0] for embedding in negative_embeddings]
    repeat_similarity_scores = [cosine_similarity(input_embedding, embedding)[0][0] for embedding in repeat_embeddings]
    
    top5avg_positive_similarity = top5avg(positive_similarity_scores)
    top5avg_negative_similarity = top5avg(negative_similarity_scores)
    top5avg_repeat_similarity = top5avg(repeat_similarity_scores)

    print(f"top5avg_positive_similarity:{top5avg_positive_similarity}, top5avg_negative_similarity:{top5avg_negative_similarity}, top5avg_repeat_similarity:{top5avg_repeat_similarity}")

    # Check which category has higher similarity
    if (top5avg_positive_similarity >= top5avg_negative_similarity) and (top5avg_positive_similarity >= top5avg_repeat_similarity):
        return 0
    elif (top5avg_repeat_similarity >= top5avg_positive_similarity) and (top5avg_repeat_similarity >= top5avg_negative_similarity):
        return 1
    else:
        return 2

positive_embeds, negative_embeds, repeat_embeds = generate_response_embeddings()

