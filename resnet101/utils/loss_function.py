import tensorflow as tf

def cosine_similarity(features):
    """
    Computes the cosine similarity for pairs of examples in the features tensor.
    Args:
        features: Tensor of shape (2N, d), where N is the number of examples and d is the feature dimension.
    Returns:
        Cosine similarity matrix of shape (2N, 2N).
    """
    # Normalize each vector in the batch (along the feature dimension)
    normalized_features = tf.nn.l2_normalize(features, axis=1)

    # Compute cosine similarity between all pairs in the batch
    similarity_matrix = tf.matmul(normalized_features, normalized_features, transpose_b=True)

    return similarity_matrix

def nt_xent_loss(features, temperature):
    """
    Computes the NT-Xent loss (Normalized Temperature-scaled Cross Entropy Loss).
    Args:
        features: Tensor of shape (2N, d), representing 2N augmented views from N examples.
        temperature: A float representing the temperature parameter.
    Returns:
        Scalar tensor representing the NT-Xent loss.
    """
    # Compute cosine similarity
    sim_matrix = cosine_similarity(features)

    # Number of examples in the mini-batch (2N)
    num_examples = tf.shape(features)[0]

    # Create a mask for excluding the positive example itself
    mask = tf.one_hot(tf.range(num_examples), num_examples)
    mask = tf.logical_not(tf.cast(mask, tf.bool))

    # Compute softmax denominator, excluding the positive example itself
    sim_matrix = sim_matrix / temperature
    exp_sim_matrix = tf.exp(sim_matrix) * tf.cast(mask, tf.float32)

    # Sum over all negative examples for each example in the batch
    sum_exp_sim_matrix = tf.reduce_sum(exp_sim_matrix, axis=1)

    # Extract the similarity of the positive pairs
    # Positive pairs are at positions (0, N), (1, N+1), ..., (N-1, 2N-1)
    positive_sim = tf.exp(tf.linalg.diag_part(sim_matrix)[::2] / temperature)

    # Compute loss for each example in the batch
    losses = -tf.math.log(positive_sim / sum_exp_sim_matrix[::2])

    # Average loss over the mini-batch
    loss = tf.reduce_mean(losses)

    return loss