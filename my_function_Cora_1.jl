########################################################## My function with Cora ##########################################################


#using Plots
using Distributions
using StatsBase
using InvertedIndices
using Flux 
using OneHotArrays
using Graphs
using GraphPlot
using Distributions
using LinearAlgebra
using Random 
using InvertedIndices
using MLDatasets: Cora



#convert adj to S marix
  function A2S(AdjMat)
    AdjMat += I #add the identity to the diagonal, add self-loops
    diag = Diagonal(vec(sum(AdjMat,dims=2) .^ (-1/2)))
    return (diag) * AdjMat * (diag) #return the normalized S matrix
end



  ###########################################################################################################################################
 
 #################################################### Functions used when parameter = 1 ###################################################
  
  #parameter = 1 means that: We drop the last n rows and columns of the adjacency matrix. Then the new S matrix is calculated. Same number of rows from x matrix and Y matrix are also dropped.
  
  
  ##This function reduces the size of a matrix by dropping the last n_col_row_to_reduce rows-columns. 
  #It uses the function A2S in order to convert the reduced adjacency matrix to S matrix. 
  #It returns the final matrix in onecold form
  #adj = aj matrix
  #x = xmatrix 
  #th = th_matrix, wheight matrix 
  #n_col_row_to_reduce = how many rows-columns we want to drop
  
  function reduce_size(adj, x, y,  n_col_row_to_reduce)
    x_dim = size(adj, 1)
    y_dim = size(adj, 2)
    k = 3
    adj_reduced = adj[Not(x_dim - n_col_row_to_reduce + 1: end), Not(y_dim - n_col_row_to_reduce + 1: end)]
    s_reduced = A2S(adj_reduced)
    x_reduced_ = x[Not(x_dim - n_col_row_to_reduce + 1: end), :]
    y_reduced_ = y[Not(x_dim - n_col_row_to_reduce + 1: end), :]
    SX_reduced_ = s_reduced^k * x_reduced_
    return SX_reduced_, y_reduced_, x_reduced_
    end

#This matrix calculates the accuracy between two matrices, the actual Ydata and the predicted_Ydata
function check_accuracy(Ydata, Ydata_predicted)
  s = 0
  for i in range(1, length(Ydata_predicted))
    if Ydata[i] == Ydata_predicted[i]
      s += 1
    end 
  end 
  accuracy = round(s/length(Ydata_predicted), digits = 2)
  println("Accyracy to get Ydata, using parameter matrix and SX i.e. (SX * parameter_matrix): ", accuracy * 100, "%")
  return accuracy
  end  


#parameter = 2 means that: We radomly drop a p% of the adjacency matrix. If k columns are dropped from adjacency matrix, then the same k rows are dropped from X matrix and Y matrix


#This function radomly reduces the ajacency matrix by p% percent
#adj_mtrx = adjacency matrix
#x_mtrx = the x matrix
#p is the percentage by which the adj_mtrx is being reduced, takes values from 0-1

  function radomly_reduced_matrix(adj_mtrx, x_mtrx, y_mtrx, p )
  Random.seed!(3)
    k = 3
    for i in range(1, Int(round(size(adj_mtrx, 1) * p, digits = 0)))
      random_number = rand(1: size(adj_mtrx, 1))
      adj_mtrx = adj_mtrx[1:end .!= random_number, 1:end .!= random_number]
      x_mtrx = x_mtrx[1:end .!= random_number, 1:end]
      y_mtrx = y_mtrx[1:end .!= random_number, 1:end]
    end
    S_reduced = A2S(adj_mtrx)
    SX_reduced_ = S_reduced^k * x_mtrx 
    x_reduced_ = x_mtrx
    y_reduced_ = y_mtrx
    return SX_reduced_, y_reduced_, x_reduced_
  end 



#this functions creates a list of concecutive numbers that they will be dropped as rows and/or as columns
#mtrx = the matrix we want to reduce
#to_drop the number of rows or columns we want to drop
function to_drop(adj_, X_, Ydata_, to_skip)
  k = 3 # will be used for S matrix calculation
  Random.seed!(10)
  rows_drop = rand(1: size(adj_, 1) - to_skip + 1)
  for i in range(1, to_skip)
    adj_ = adj_[1:end .!= rows_drop, 1:end ]
    adj_ = adj_[1:end, 1:end .!= rows_drop]
    X_ = X_[1:end .!= rows_drop, 1:end ]
    Ydata_ = Ydata_[1:end .!= rows_drop, 1:end]
  end
  S = A2S(adj_)
  SX_reduced = S^k * X_
  return SX_reduced, Ydata_, X_
end 


#This function loads and trains the model by using one shot learning.
#SX_mtrx = The S_matrix * X_matrix
#x_matrix = The X matrix
#y_matrix = The Y matrix (the Ydata)
function load_and_train_model_one_shot_learining(SX_mtrx, x_mtrx, y_mtrx)
  #Random.seed!(201)
  testing_perc = 0.4
  targets = collect(1:7) #length(unique(y_matrix)) #collect(1:2) #2 = length(unique(y_matrix))
  error_dict = Dict()
  res_dict = Dict()
  stopInd = size(x_mtrx)[1]
  testInds = Int.(round.( range(1, stop=stopInd, length=Int(round(testing_perc*stopInd))) ))
  trainInds = (1:stopInd)[Not(testInds)]
  X_dim =size(x_mtrx')[1]
  Y_dim = length(unique(y_mtrx))
  yhot = onehotbatch( y_mtrx , collect(1:Y_dim) )
  modelSGC = Chain( Dense( X_dim => Y_dim ) , softmax )
  optim = Flux.Adam( 0.01 );
  pars = Flux.params( modelSGC )
  data = Flux.DataLoader( (SX_mtrx', yhot) ,  batchsize=size(yhot)[2]  ,  shuffle=true )
  epoch_num = 1000
  errors = Vector{Float64}()
  for epoch in 1:epoch_num 
    for (x, y) in data
      val, grad = Flux.withgradient(pars) do
        Flux.crossentropy( modelSGC(x)[:,trainInds] , y[:,trainInds] )
      end 
   append!(errors, val)
    Flux.update!(optim, pars, grad)
    end
  end
  resDict = Dict()
  resDict["errors"] = errors
  resDict["training"] = mean( onecold( modelSGC(SX_mtrx')[:,trainInds], targets ) .== y_mtrx[trainInds] )
  resDict["testing"]  = mean( onecold( modelSGC(SX_mtrx')[:,testInds], targets) .== y_mtrx[testInds] )
  resDict["model"] = modelSGC
  weight = resDict["model"].layers[1].weight
  println("Acuuracy from training data: ", round(resDict["training"] * 100, digits = 4), "%")
  println("Acuuracy from testing data: ", round(resDict["testing"] * 100, digits = 4), "%")
  return round(resDict["training"] * 100, digits = 4), round(resDict["testing"] * 100, digits = 4), weight
end



function average_the_parameters(adj_mtrx, x_mtrx, y_mtrx, method)
    k = 3
    SX_mtrx = A2S(adj_mtrx)^k * x_mtrx
    weight_dict = Dict()
    m = 1
  if method == 1 #0.24
    for i in [0.025, 0.02]
      Random.seed!(3)
      skiped = round(size(adj_mtrx, 1) * i , digits = 0)
      SX_reduced_new, y_reduced_new, x_reduced_new = reduce_size(adj_mtrx, x_mtrx, y_mtrx, skiped)
      y_reduced_new = vec(y_reduced_new)
      train, test, weight_new = load_and_train_model_one_shot_learining(SX_reduced_new, x_reduced_new, y_reduced_new)
      weight_dict["w_$m"] = weight_new
      m += 1 
    end
      weight_sum = zeros(size(weight_dict["w_1"], 1), size(weight_dict["w_1"], 2))
      for i in range(1, length(weight_dict))
        weight_sum +=  weight_dict["w_$i"]
      end 
      weight_average_matrix = weight_sum / length(weight_dict)
      SX_weigth_new = SX_mtrx * weight_average_matrix'
      predicted_mtrx_new = onecold( SX_weigth_new')
      acc = check_accuracy(y_mtrx, predicted_mtrx_new)
    elseif method == 2
      for i in [0.11, 0.2] #0.11, 0.13, 0.14, 0.16, 0.2, 0.22, 0.23 0.24
        Random.seed!(1)
        SX_reduced_new, y_reduced_new, x_reduced_new = radomly_reduced_matrix(adj_mtrx, x_mtrx, y_mtrx, i ); 
        y_reduced_new = vec(y_reduced_new)
        train, test, weight_new = load_and_train_model_one_shot_learining(SX_reduced_new, x_reduced_new, y_reduced_new)
        weight_dict["w_$m"] = weight_new
        m += 1 
      end
        weight_sum = zeros(size(weight_dict["w_1"], 1), size(weight_dict["w_1"], 2))
        for i in range(1, length(weight_dict))
          weight_sum +=  weight_dict["w_$i"]
        end 
        weight_average_matrix = weight_sum / length(weight_dict)
        SX_weigth_new = SX_mtrx * weight_average_matrix'
        predicted_mtrx_new = onecold( SX_weigth_new')
        acc = check_accuracy(y_mtrx, predicted_mtrx_new)
    elseif method == 3 # 0.14, 0.25 kalo, 0.26, 0.3 kalo, 0.32 kalo, 0.37 kalo, 0.41 kalo, 0.42 kalo, 0.43 kalo, 0.44 , 0.47, 0.48   
        for i in [0.25, 0.41] 
      Random.seed!(1)
      skiped = round(size(adj_mtrx, 1) * i , digits = 0)
      SX_reduced_new, y_reduced_new, x_reduced_new = to_drop(adj_mtrx, x_mtrx, y_mtrx, skiped)
      y_reduced_new = vec(y_reduced_new)
      train, test, weight_new = load_and_train_model_one_shot_learining(SX_reduced_new, x_reduced_new, y_reduced_new)
      weight_dict["w_$m"] = weight_new
      m += 1 
    end
      weight_sum = zeros(size(weight_dict["w_1"], 1), size(weight_dict["w_1"], 2))
      for i in range(1, length(weight_dict))
        weight_sum +=  weight_dict["w_$i"]
      end 
      weight_average_matrix = weight_sum / length(weight_dict)
      SX_weigth_new = SX_mtrx * weight_average_matrix'
      predicted_mtrx_new = onecold( SX_weigth_new')
      acc = check_accuracy(y_mtrx, predicted_mtrx_new)
  end 
end 

function my_function(adj_mtrx, x_mtrx, y_mtrx, parameter)
    k = 3
    SX_mtrx = A2S(adj_mtrx)^k * x_mtrx
    weight_dict = Dict()
    if parameter == 1
      Random.seed!(3)
      a = 0.26 #droping the a% of the matrix
      skiped = round(size(adj_mtrx, 1) * a , digits = 0)
      SX_reduced, y_reduced, x_reduced = reduce_size(adj_mtrx, x_mtrx, y_mtrx, skiped)
      y_reduced = vec(y_reduced)
      train, test, weight = load_and_train_model_one_shot_learining(SX_reduced, x_reduced, y_reduced)
      SX_weigth = SX_mtrx * weight'
      predicted_mtrx = onecold( SX_weigth' )
      acc = check_accuracy(y_mtrx, predicted_mtrx)
      println(repeat("-", 48), " Calculating the average Parameter ", repeat("-", 48))
      average_the_parameters(adj_mtrx, x_mtrx, y_mtrx, parameter)
    elseif parameter == 2 
      Random.seed!(1)
      a = 0.3 #droping the a% of the matrix
      SX_reduced, y_reduced, x_reduced = radomly_reduced_matrix(adj_mtrx, x_mtrx, y_mtrx, a ); 
      y_reduced = vec(y_reduced)
      train, test, weight = load_and_train_model_one_shot_learining(SX_reduced, x_reduced, y_reduced)
      SX_weigth = SX_mtrx * weight'
      predicted_mtrx = onecold( SX_weigth' )
      acc = check_accuracy(y_mtrx, predicted_mtrx)
      println(repeat("-", 48), " Calculating the average Parameter ", repeat("-", 48))
      average_the_parameters(adj_mtrx, x_mtrx, y_mtrx, parameter)
    elseif parameter == 3
      Random.seed!(1)
      a = 0.3 #droping the a% of the matrix 0.25
      skiped = round(size(adj_mtrx, 1) * a , digits = 0)
      SX_reduced, y_reduced, x_reduced= to_drop(adj_mtrx, x_mtrx, y_mtrx, skiped)
      y_reduced = vec(y_reduced)
      train, test, weight  = load_and_train_model_one_shot_learining(SX_reduced, x_reduced, y_reduced)
      SX_weigth = SX_mtrx * weight'
      predicted_mtrx = onecold( SX_weigth' )
      acc = check_accuracy(y_mtrx, predicted_mtrx)
      println(repeat("-", 48), " Calculating the average Parameter ", repeat("-", 48))
      average_the_parameters(adj_mtrx, x_mtrx, y_mtrx, parameter)
    end
   # return values
  #return  train, test, weight
end 


data = Cora.dataset()
keys(Cora.dataset())


function adj_list_to_matrix(adj_list)
  n = length(adj_list)
  mtrx = zeros(n, n)
  #row_number counts in which row we are in the vector element
  row_number = 0
  for i in range(1, n)
    for j in adj_list[i]    
      mtrx[i, j] = 1
      #println("($i, $j, 1)")
    end 
  end 
  return mtrx
end 

adj_cora = adj_list_to_matrix(data.adjacency_list);
X_cora = data.node_features;
Ydata_cora = data.node_labels;


println(repeat("-", 60), " Cora Dataset ", repeat("-", 60))
println(repeat("-", 54), " Description of what I did ", repeat("-", 54))
println("\n")
println("Reduce Matrices ==> Train model(one shot learning) ==> Get parameters of reduced with trained data ==> ")
println("Back to unreduced data with accuracy a% ==> Method to calculate unreduced data with accuracy higher than a%")
println("Method: (Reduce the unreduced matrices by a certain percent ==> get the parameters and average them ==> ")
println("Back to unreduced data with accuracy a_new%  where a_new% > a%)")
println("\n")


for i in 1:3
  println(repeat("-", 60), " Method $i ", repeat("-", 60))
  v = my_function(adj_cora, X_cora', Ydata_cora, i)
end 


