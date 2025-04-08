# COO-SPMV with Spark

### Results
Spark was tested locally with python, so the performance is less than ideal. I also ran into memory errors with larger matrices, so a block solution was used to parse the data in parts. Spark was significantly slower than 

```
def spmv_coo_spark(spark: SparkSession,
				   coo_matrix: sparse.coo_matrix,
				   x_vector: np.ndarray) -> np.ndarray:	
	# Parallelize COO matrix to RDD of (row, col, val)
	matrix_rdd: RDD = spark.sparkContext.parallelize(
		zip(coo_matrix.row, coo_matrix.col, coo_matrix.data)
	)
	
	# Broadcast the vector x
	x_broadcast = spark.sparkContext.broadcast(x_vector)
	
	time_start = time.time()
	partial_rdd = matrix_rdd.map(lambda x: (x[0], x[2] * x_broadcast.value[x[1]]))
	result_rdd: List[Tuple[int, float]] = partial_rdd.reduceByKey(lambda a, b: a + b)\
										  .collect()

	time_end = time.time()
	print(f"Spark SpMV time: {time_end - time_start:.4f} seconds")
	
	# cleanup
	x_broadcast.unpersist()

	# Collect results into a dense array
	result: np.ndarray = np.zeros(coo_matrix.shape[0])
	for row_id, value in result_rdd:
		result[row_id] = value
	
	return result
```