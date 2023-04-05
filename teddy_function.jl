using Plots
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

##################################################### generate data to test the my_function ######################################################


#convert adj to S marix
  function A2S(AdjMat)
    AdjMat += I #add the identity to the diagonal, add self-loops
    diag = Diagonal(vec(sum(AdjMat,dims=2) .^ (-1/2)))
    return (diag) * AdjMat * (diag) #return the normalized S matrix
end

#This function generates the Dataset

function AdjMat_XMat_YMat()
  Random.seed!(10)
  nn_max = 3
  NN_set = 10^nn_max
  NN_tmp = Int( NN_set / 4 )
  ER_tmp = erdos_renyi( NN_tmp , 10*(NN_tmp) )
  BA_tmp = barabasi_albert( NN_tmp , 8 )
  SF_tmp = static_scale_free( NN_tmp , 8*(NN_tmp) , 4 )
  WS_tmp = erdos_renyi( NN_tmp , 10*(NN_tmp) ) #barabasi_albert( NN_tmp , 5 )
  blocks_tmp = blockdiag( blockdiag( blockdiag(ER_tmp,BA_tmp),SF_tmp ), WS_tmp )
    
  ER_BA_SF_WS_Block_graphs = blockdiag( blockdiag( blockdiag(ER_tmp,BA_tmp),SF_tmp ), WS_tmp )
  #now add some edges between the blocks that are the communities
  for bb in 1:Int(round(NN_tmp/10))
    for b1 in 0:3
      for b2 in 0:3
        if(b1 < b2)
          range1 = randperm(NN_tmp)[1] + b1*NN_tmp
          range2 = randperm(NN_tmp)[1] + b2*NN_tmp
          add_edge!( ER_BA_SF_WS_Block_graphs , range1 , range2 )
        end
      end
    end
  end 
  #density_nn = Graphs.density(ER_BA_SF_WS_Block_graphs)
  ER_BA_SF_WS_Block_matrices = Matrix(adjacency_matrix(ER_BA_SF_WS_Block_graphs))

  d1 = Dirichlet( [10,10,10] )
  c1 = Categorical( [0.5,0.25,0.25] )
  d2 = Dirichlet( [20,10,10] )
  c2 = Categorical( [0.35,0.35,0.3] )
  d3 = Dirichlet( [20,10,20] )
  c3 = Categorical( [0.25,0.25,0.5] )

  networks_X = Dict()
  networks_Y = Dict()
  networks_Y_cold = Dict()

    
    NN_tmp = Int( NN_set / 4 )
    
    xd1 = rand( d1 , NN_tmp )
    xc1 = onehotbatch( rand( c1 , NN_tmp ) , 1:3 )
    x1a = vcat( xd1 , xc1 )'
    xd1 = rand( d1 , NN_tmp )
    xc1 = onehotbatch( rand( c1 , NN_tmp ) , 1:3 )
    x1b = vcat( xd1 , xc1 )'
    xd2 = rand( d2 , NN_tmp )
    xc2 = onehotbatch( rand( c2 , NN_tmp ) , 1:3 )
    x2 = vcat( xd2 , xc2 )'
    xd3 = rand( d3 , NN_tmp )
    xc3 = onehotbatch( rand( c3 , NN_tmp ) , 1:3 )
    x3 = vcat( xd3 , xc3 )'
    xc3 = onehotbatch( rand( c3 , NN_tmp ) , 1:3 )
    x3 = vcat( xd3 , xc3 )'

    networks_X = vcat( x1a , x1b , x2 , x3 )
    XMat = networks_X
    
    y1a = onehotbatch( 1*ones(NN_tmp) , 1:2 )'
    y1b = onehotbatch( 1*ones(NN_tmp) , 1:2 )'
    y2 = onehotbatch( 2*ones(NN_tmp) , 1:2 )'
    y3 = onehotbatch( 2*ones(NN_tmp) , 1:2 )'
    
    networks_Y = vcat( y1a , y1b , y2 , y3 )
    networks_Y_cold = vcat(1*ones(NN_tmp),1*ones(NN_tmp),2*ones(NN_tmp),2*ones(NN_tmp))       
    YMat = networks_Y_cold
    return ER_BA_SF_WS_Block_matrices, XMat, YMat
end 

 adj_matrix, x_matrix, y_matrix  = AdjMat_XMat_YMat();
 
 ##################################################################################################################################################
 #################################################### Functions used when parameter = 1 ####################################################
  
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
  end  


#parameter = 2 means that: We radomly drop a p% of the adjacency matrix. If k columns are dropped from adjacency matrix, then the same k rows are dropped from X matrix and Y matrix


#This function radomly reduces the ajacency matrix by p% percent
#adj_mtrx = adjacency matrix
#x_mtrx = the x matrix
#p is the percentage by which the adj_mtrx is being reduced, takes values from 0-1

  function radomly_reduced_matrix(adj_mtrx, x_mtrx, y_mtrx, p )
    k = 3
    for i in range(1, Int(size(adj_mtrx, 1) * p))
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
function load_and_train_model_one_shot_learining(SX_mtrx, x_matrix, y_matrix)
  Random.seed!(2)
  testing_perc = 0.4
  targets = collect(1:2) #length(unique(y_matrix)) #collect(1:2) #2 = length(unique(y_matrix))
  error_dict = Dict()
  res_dict = Dict()
  stopInd = size(x_matrix)[1]
  testInds = Int.(round.( range(1, stop=stopInd, length=Int(round(testing_perc*stopInd))) ))
  trainInds = (1:stopInd)[Not(testInds)]
  X_dim =size(x_matrix')[1]
  Y_dim = length(unique(y_matrix))
  yhot = onehotbatch( y_matrix , collect(1:Y_dim) )
  modelSGC = Chain( Dense( X_dim => Y_dim ) , softmax )
  optim = Flux.Adam( 0.01 );
  pars = Flux.params( modelSGC )
  data = Flux.DataLoader( (SX_mtrx', yhot) ,  batchsize=size(yhot)[2]  ,  shuffle=true )
  epoch_num = 1000
  errors = []
  for epoch in 1:epoch_num 
    for (x, y) in data
      val, grad = Flux.withgradient(pars) do
        Flux.crossentropy( modelSGC(x)[:,trainInds] , y[:,trainInds] )
      end 
    push!(errors, val)
    Flux.update!(optim, pars, grad)
    end
  end
  resDict = Dict()
  resDict["errors"] = errors
  resDict["training"] = mean( onecold( modelSGC(SX_mtrx')[:,trainInds], targets ) .== y_matrix[trainInds] )
  resDict["testing"]  = mean( onecold( modelSGC(SX_mtrx')[:,testInds], targets) .== y_matrix[testInds] )
  resDict["model"] = modelSGC
  weight = resDict["model"].layers[1].weight
  SX_weigth = SX_mtrx * weight'
  predicted_mtrx = onecold( SX_weigth' )
  println("Acuuracy from training data: ", round(resDict["training"] * 100, digits = 4), "%")
  println("Acuuracy from testing data: ", round(resDict["testing"] * 100, digits = 4), "%")
  check_accuracy(y_matrix, predicted_mtrx)
end 

 ################################################################## my_function ##################################################################

  #adj_matrix = adjacency matrix
  #x_matrix = x matrix
  #y_matrix = y matrix (actual data)
  #paremeter = says which method will be used to split the adj_matrix 
  
  function my_function(adj_matrix, x_matrix, y_matrix, parameter)
    if parameter == 1
      a = 0.4 #droping the a% of the matrix
      skiped = round(size(adj_matrix, 1) * a , digits = 0)
      SX_reduced, y_reduced, x_reduced = reduce_size(adj_matrix ,x_matrix, y_matrix, skiped)
      y_reduced = vec(y_reduced)
      load_and_train_model_one_shot_learining(SX_reduced, x_reduced, y_reduced)
    elseif parameter == 2 
      SX_reduced, y_reduced, x_reduced = radomly_reduced_matrix(adj_matrix ,x_matrix, y_matrix, 0.4 ); 
      y_reduced = vec(y_reduced)
      load_and_train_model_one_shot_learining(SX_reduced, x_reduced, y_reduced)
    elseif parameter == 3
      a = 0.25 #droping the a% of the matrix 0.25
      skiped = round(size(adj_matrix, 1) * a , digits = 0)
      SX_reduced, y_reduced, x_reduced= to_drop(adj_matrix, x_matrix, y_matrix, skiped)
      y_reduced = vec(y_reduced)
      load_and_train_model_one_shot_learining(SX_reduced, x_reduced, y_reduced)
    end 
  #return  accuracy, epochs, loss, predicted_mtrx
end 

##################################################################################################################################################
########################################################### Apply the function to data ###########################################################

for i in 1:3
    my_function(adj_matrix, x_matrix, y_matrix, i)
    println("*" * repeat("-", 100))
  end
