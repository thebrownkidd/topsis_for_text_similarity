from sentence_transformers import SentenceTransformer,util

def semantic_similarity(s1,s2):
    model =  SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    em1 = model.encode(s1, convert_to_tensor = True)
    em2 = model.encode(s2, convert_to_tensor = True)

    sim =  util.pytorch_cos_sim(em1, em2)

    return(sim[0][0])
