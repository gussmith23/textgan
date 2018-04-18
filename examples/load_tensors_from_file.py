import tensorflow as tf


embeddings = tf.Variable(-1.0, validate_shape=False, name="normalized_embeddings")
saver = tf.train.Saver()

with tf.Session() as sess:
  saver.restore(sess, "log/embeddings_model.ckpt")
  out_embeddings = sess.run(embeddings)
  
print(out_embeddings)
print(out_embeddings.shape)

graph = tf.Graph()
with graph.as_default(): 
  embeddings = tf.Variable(-1.0, validate_shape=False, name="normalized_embeddings")
  init = tf.global_variables_initializer()
  saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
  init.run()
  print(sess.run([embeddings]))