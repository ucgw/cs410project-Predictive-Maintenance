from gensim import corpora
from gensim.models import LdaModel, LsiModel
from gensim.models.coherencemodel import CoherenceModel


def lda(metadata, n_topics, new_tokens):
    """
    Perform Latent Dirichlet Allocation (LDA) on the given metadata.

    Parameters:
    - metadata (list): Metadata
    - n_topics (int): The number of topics to discover in metadata
    - new_tokens (list): Tokenized representation of a new request for topic prediction.

    Returns:
    tuple: A tuple containing three elements:
        1. lda_pi (list): Topic probability distribution for the new maintenance request.
        2. lda_P (list): Probabilities of words given topics.
        3. coherence_scores (dict): Coherence scores for different metrics (u_mass, c_v, c_uci, c_npmi).
    """
    tokens = [d["tokens"] for elem in metadata for d in elem]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    new_token_corpus = dictionary.doc2bow(new_tokens)

    lda_model = LdaModel(corpus, num_topics=n_topics, id2word=dictionary)
    lda_pi = lda_model[new_token_corpus]  # get topic probability distribution for a new maintenance request
    lda_pi = [elem[1] for elem in lda_pi]
    lda_P = lda_model.get_topics()  # probabilities of words given topics

    coherence_scores = get_coherence_scores(lda_model, tokens, dictionary)

    return lda_pi, lda_P, coherence_scores


def lsa(metadata, n_topics, new_tokens):
    """
    Perform Latent Semantic Analysis (LSA) on the given metadata.

    Parameters:
    - metadata (list): Metadata
    - n_topics (int): The number of topics to discover in metadata
    - new_tokens (list): Tokenized representation of a new maintenance request

    Returns:
    tuple: A tuple containing two elements:
        1. term_contributions (list): List of words defining the dominant topic and their contributions.
        2. coherence_scores (dict): Coherence scores for different metrics (u_mass, c_v, c_uci, c_npmi).
    """
    tokens = [d["tokens"] for elem in metadata for d in elem]
    dictionary = corpora.Dictionary(tokens)
    # Converting list of documents into Document Term Matrix using dictionary prepared above
    doc_term_matrix = [dictionary.doc2bow(token) for token in tokens]

    lsa_model = LsiModel(doc_term_matrix, num_topics=n_topics, id2word=dictionary)

    new_token_corpus = dictionary.doc2bow(new_tokens)

    lsa_output = lsa_model[new_token_corpus]  # get the underlying topics coefficients for the new maintenance request
    # Find the dominant topic - the element with the greatest absolute value
    main_topic_number, _ = max(lsa_output, key=lambda x: abs(x[1]))

    # Get the list of words that define the dominant topic along with their contribution
    term_contributions = lsa_model.get_topics()[main_topic_number]  # lsa_model.show_topic(main_topic_number)

    coherence_scores = get_coherence_scores(lsa_model, tokens, dictionary)

    return term_contributions, coherence_scores


def get_coherence_scores(model, tokens, dictionary):
    """
    Calculate coherence scores for a given topic modeling model.

    Parameters:
    - model: The topic modeling model (LDA or LSA).
    - tokens (list): Tokenized representation of documents.
    - dictionary: Gensim dictionary object.

    Returns:
    dict: Coherence scores for different metrics (u_mass, c_v, c_uci, c_npmi).
    """
    coherence_metrics = ['u_mass', 'c_v', 'c_uci', 'c_npmi']
    coherence_scores = {}

    for coherence_metric in coherence_metrics:
        coherence_model = CoherenceModel(model=model, texts=tokens, dictionary=dictionary,
                                         coherence=coherence_metric)
        score = coherence_model.get_coherence()
        coherence_scores[coherence_metric] = score
    return coherence_scores
