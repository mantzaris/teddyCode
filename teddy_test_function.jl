######################################################## building the my_function file ########################################################

##This function reduces the size of a matrix by throughing the last n_col_row_to_reduce rows-columns
#adj = aj matrix
#x = xmatrix 
#th = th_matrix, wheight matrix 
#n_col_row_to_reduce = how many rows-columns we want to drop

function reduce_size(adj, x, th, n_col_row_to_reduce)
  x_dim = size(adj, 1)
  y_dim = size(adj, 2)
  k = 3
  adj_reduced = adj[Not(x_dim - n_col_row_to_reduce + 1: end), Not(y_dim - n_col_row_to_reduce + 1: end)]
  s_reduced = A2S(adj_reduced)
  x_reduced = x[Not(x_dim - n_col_row_to_reduce + 1: end), :]
  Final_reduced = s_reduced^k * x_reduced * th'
  return onecold(Final_reduced')
  end

#This function loads the data and trains the neural network. 
#SX_ = the S*X matrix (X data)
#yhot = the Y data with (onehotbatch)

function load_and_train_model(SX_, yhot_)
  resDict_ = Dict()
  model_ = Chain( Dense( size(SX_, 1) => size(yhot_, 1)) , softmax)
  loss(x, y) = Flux.crossentropy(model_(x), y)
  opt = Adam(0.01)
  pars = Flux.params(model_)
  data = Flux.DataLoader((SX_, yhot_) , batchsize = 10 , shuffle = true)
  epochs_ = Int64[]
  for epoch in 1:500
    Flux.train!(loss, pars, data ,opt)
    push!(epochs_, epoch)
  end 
  resDict_["params"] = pars
  resDict_["model"] = model_
  return(model_, resDict_)
end 



#adj_matrix = adjacency matrix
#x_matrix = x matrix
#y_matrix = y matrix (actual data)
#paremeter = says which method will be used to split the adj_matrix 

function my_function(adj_matrix, x_matrix, y_matrix, parameter)
  #get Y' data by training the network, before seting the parameter
  k = 3
  yhot = onehotbatch(y_matrix, [1, 2])
  S = A2S(adj_matrix)
  SX = S^k * x_matrix
  SX = SX'   
  train_x = SX
  train_y = yhot
  model, resDict = load_and_train_model(SX, yhot)
  weight = resDict["model"].layers[1].weight
  accuracy = round(mean( onecold( model(train_x), [1, 2] ) .== onecold(train_y, [1, 2]) ) * 100, digits = 2)
  println("Accuracy by training all data: ", accuracy, "%", ", k = $k")  
  #test = weight * SX
  #SX_TH = onecold(test, [1, 2])
  if parameter == 1
    skiped = round(size(adj_matrix, 1) * 0.25, digits = 0)
    f_reduced = reduce_size(adj_matrix ,x_matrix, weight, skiped)
  end 
  return f_reduced
end 
