import tensorflow as tf

def meanDistance(y_true, y_pred):
    in_shape = tf.shape(y_true)

    # Flatten height/width dims
    flat_true = tf.reshape(y_true, [in_shape[0], -1, in_shape[-1]])
    flat_pred = tf.reshape(y_pred, [in_shape[0], -1, in_shape[-1]])

    # Find peaks in linear indices
    idx_true = tf.argmax(flat_true, axis=1)
    idx_pred = tf.argmax(flat_pred, axis=1)

    # Convert linear indices to subscripts
    rows_true = tf.floor_div(tf.cast(idx_true,tf.int32), in_shape[2])
    cols_true = tf.floormod(tf.cast(idx_true,tf.int32), in_shape[2])
    
    rows_pred = tf.floor_div(tf.cast(idx_pred, tf.int32), in_shape[2])
    cols_pred = tf.floormod(tf.cast(idx_pred, tf.int32), in_shape[2])
    
    row_diff = tf.square(tf.subtract(tf.cast(rows_true, tf.float32), tf.cast(rows_pred, tf.float32)))
    col_diff = tf.square(tf.subtract(tf.cast(cols_true, tf.float32), tf.cast(cols_pred, tf.float32)))
    distances = tf.sqrt(tf.add(row_diff, col_diff))
    
    return tf.reduce_mean(distances)