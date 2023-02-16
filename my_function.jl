################################# Functions used for all parameters in the function called my_function() #################################



function split_train_test(Xdata, Ydata, percentage_to_train)
    #sample function gets numbers without repeating them
    number_of_rows = Int(round(percentage_to_train * size(Xdata, 1), digits = 0))
    println(number_of_rows)
    println(typeof(number_of_rows))
    train_id = sort(sample(1: size(Xdata, 1), number_of_rows, replace = false))
    test_id = (1:size(Xdata, 1))[Not(train_id)]
    trainX = Xdata[train_id, :]
    testX = Xdata[test_id, :]
    trainY = Ydata[train_id, :]
    testY = Ydata[test_id, :]
    return trainX, testX, trainY, testY
  end
  
  
  
  #This function loads the data and trains the neural network. 
  #SX_ = the S*X matrix (X data)
  #yhot = the Y data with (onehotbatch)
  #This function is used in the function called my_function(), in order to load and train the data of the neural network
  #it returns the model and resDict 
  function load_and_train_model(SX_, yhot_)
    resDict_ = Dict()
    model_ = Chain( Dense( size(SX_, 1) => size(yhot_, 1)) , softmax)
    loss(x, y) = Flux.crossentropy(model_(x), y)
    opt = Adam(0.01)
    pars = Flux.params(model_)
    data = Flux.DataLoader((SX_, yhot_) , batchsize = 10 , shuffle = true)
    epochs_ = Int64[]
    loss_ = Float64[]
    for epoch in 1:500
      Flux.train!(loss, pars, data ,opt)
      push!(epochs_, epoch)
      push!(loss_, loss(SX_, yhot_))
    end 
    resDict_["params"] = pars
    resDict_["model"] = model_
    return(model_, resDict_, epochs_, loss_)
  end 




  #################################################### Functions used when parameter = 1 ####################################################
  
  #parameter = 1 means that: We drop the last n rows and columns of the adjacency matrix. Then the new S matrix is calculated. Same number of rows from x matrix are also dropped.
  
  
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
    x_reduced = x[Not(x_dim - n_col_row_to_reduce + 1: end), :]
    y_reduced_ = y[Not(x_dim - n_col_row_to_reduce + 1: end), :]
    SX_reduced_ = s_reduced^k * x_reduced
    return SX_reduced_, y_reduced_
    end

#This matrix calculates the accuracy between two matrices, the actual Ydata and the predicted_Ydata
function check_accuracy(Ydata, Ydata_predicted)
  s = 0
  for i in range(1, length(Ydata_predicted))
    if Ydata[i] == Ydata_predicted[i]
      s += 1
    end 
  end 
  accuracy = s/length(Ydata_predicted)
  println("Accyracy by testing the big matrix is: ", accuracy * 100, "%")
  end  


  
#################################################### Functions used when parameter = 2 ####################################################
  #parameter = 2 means that: adjacency matrix is splited in smaller matrices with size (size(adj_mtrx, 1)/ n_lines_to_split, size(adj_mtrx, 1)/ n_lines_to_split) and each smaller #matrix is saved in a dictionary. Then the x matrix is splited accordingly and each smaller matrix is saved in another dictionary.
  #Then the dictionary with the splited adjacency matrix is converted to S matrix and then is multiplied with the dictionary containing the #splited x matrices. Then this dictionary is converted to matrix and multiplied with weight function
  
  
  
  #This function gets the divisors of a number
  #It is used when parameter = 2 in order to get the n_lines_to_split (mentioned on the above), in order to split the adjacency matrix. Code #uses get_divisors(size(adj_matrix, 2))[2] because 1 gives the original adjacency matrix, so 2 is the next one. We can definately use other numbers such as 3 or 4. Example, if adjacency matrix is 10 x 10 then get_divisors() will return 1 2 5 10. If [1] is used, then adj_matrix will break in  10/1 x 10/1 = 10 x10 matrix (same as previousely). If [2] is selected, then adj_matrix will break as  10/2 x 10/2 = 5 x 5 matrix. So the adj_matrix will be splited in four 5x5 matrices.
  
  function get_divisors(number)
    divisors = Vector{Float64}()
    for i in range(1, div(number, 2))
      if number % i == 0
        append!(divisors, Int(i))
      end
    end
    append!(divisors, Int(number))
    return divisors
  end
  
  
  #This function splits the adjacency matrix  matrix in smaller matrices which size is (n_lines_to_split x n_lines_to_split). 
  #adj_mtrx = the adj matrix
  #size(adj_mtrx, 1)/ n_lines_to_split is the number of rows, columns the small matrix will have.
  #The function returns a dictionary. Each key element of the dictionary is one of the smaller matrices.
  #This function is used in the function my_function() in when the parameter is equal to 2. 
  #It returns the adjacency matrix splited as a dictionary
  #it is used in the function multiply_splited_2() in order to split the adjacency matrix which is given
  
   function split_adj_matrix(adj_mtrx, n_lines_to_split)
    x = size(adj_mtrx, 1)
    y = size(adj_mtrx, 2)
    n = div(x, n_lines_to_split)
    dictionary = Dict()
    count = 1
    k = 1
    for i in range(1, n_lines_to_split)
      z = 1
      for j in range(1, n_lines_to_split)
        dictionary["a_$count"] = adj_mtrx[k: k + n - 1, z: j*n ]
        count = count + 1
        z = z + n
      end
      k = k + n
    end
    return(dictionary)
  end
  
  
  #This function splits the x matrix in smaller matrices which size is (n_lines_to_split x n_lines_to_split). 
   #a_mtrx = the a_matrix
   #size(adj_mtrx, 1)/ n_lines_to_split is the number of rows, columns the small matrix will have.
   #The function returns a dictionary. Each key element of the dictionary is one of the smaller matrices.
   #This function is used in the function my_function() in when the parameter is equal to 2. 
   #It is the same with the function split_adj_matrix() but it splits the x matix
   #It returns the x matrix splited as a dictionary
   #it is used in the function multiply_splited_2() in order to split the x matrix which is given
    function split_x_matrix(x_mtrx, n_lines_to_split)
    x = size(x_mtrx, 1)
    n = div(x, n_lines_to_split)
    #print(n)
    dictionary = Dict()
    z = 1
    for i in range(1, n_lines_to_split)
      dictionary["x_$i"] = x_mtrx[z : z + n - 1 , 1:end]
      z = z + n
    end
    return(dictionary)
    end
  

  #This function splits the y matrix in smaller matrices which size is (n_lines_to_split x n_lines_to_split). 
   #y_mtrx = the Ydata
   #size(adj_mtrx, 1)/ n_lines_to_split is the number of rows, columns the small matrix will have.
   #The function returns a dictionary. Each key element of the dictionary is one of the smaller matrices.
   #This function is used in the function my_function() in when the parameter is equal to 2. 
   #It is the same with the function split_adj_matrix() but it splits the y matix
   #It returns the y matrix splited as a dictionary
   #it is used in the function multiply_splited_2() in order to split the Ydata matrix which is given
    function split_y_matrix(y_mtrx, n_lines_to_split)
    x = size(y_mtrx, 1)
    n = div(x, n_lines_to_split)
    print(n)
    dictionary = Dict()
    z = 1
    for i in range(1, n_lines_to_split)
      dictionary["y_$i"] = y_mtrx[z : z + n - 1 , 1:end]
      z = z + n
    end
    return(dictionary)
    end

  
  #This function creates the adjacency matrix times the x matrix in dictionary form
  #a_mtrx = adjacency matrix
  #x_mtrx = X matrix 
  #size(adj_mtrx, 1)/ n_lines_to_split is the number of rows, columns the small matrix will have.

  function adj_matrxix_x_X_matrix_dictionary(a_mtrx, x_mtrx, n_lines_to_split)
  k = 3
  a_splited = split_adj_matrix(a_mtrx, n_lines_to_split)
  x_splited = split_x_matrix(x_mtrx, n_lines_to_split)
  #x_y_dim = size(x_mtrx, 2)
  aj_splited = Dict()
  for i in range(1, length(a_splited))
    aj_splited["aj_$i"] = A2S(a_splited["a_$i"])
  end
  sx = Dict()
  k = 1
  while k <= length(a_splited)
    for i in range(1, length(x_splited))
      sx["sx_$k"] = aj_splited["aj_$k"]^3 * x_splited["x_$i"] 
      k = k + 1
    end
  end
  return sx 
  end 
  
  

#This function combines the data we have. For example if SX_segmented = {"SX_1", "SX_2", "SX_3", "SX_4", "SX_5", "SX_6", "SX_7", "SX_8", "SX_9", "SX_10"} 
#and Ydata_segmented = {"Y1", "Y2"}, then this function returns this:
#[(SX_1, Y1), (SX_2, Y1), (SX_3, Y1), (SX_4, Y1), (SX_5, Y1), (SX_6, Y2), (SX_7, Y2), (SX_8, Y2), (SX_9, Y2), (SX_10, Y2) ], this if done for training purposes
#SX_segmented is a dictionary. SX_1, SX_2....are smaller matrices, so SX_segmented can be converted to a matrix [SX_1, SX_2, SX_3, SX_4, SX_5; SX_6, SX_7, SX_8, SX_9, SX_10] which is the original SX matrix

#SX_segmented_ = the segmented SX matrix in dictionary form
#Ydata_segmented_ = the segmented Ydata matrix in dictionary form
function combine_data(SX_segmented_, Ydata_segmented_)
  combined_data_ = []
  count = 1
  for i in range(1, length(Ydata_segmented_))
    for j in range(1, length(SX_segmented_) / length(Ydata_segmented_))
      push!(combined_data_, (SX_segmented_["sx_$count"], Ydata_segmented_["y_$i"]))
      count += 1
    end 
  end 
  return combined_data_ 
end 


  function radomly_reduced_matrix(adj_mtrx, x_mtrx, y_mtrx )
    k = 3
    for i in range(1, Int(size(adj_mtrx, 1) * 0.1))
      random_number = rand(1: size(adj_mtrx, 1))
      adj_mtrx = adj_mtrx[1:end .!= random_number, 1:end .!= random_number]
      x_mtrx = x_mtrx[1:end .!= random_number, 1:end]
      y_mtrx = y_mtrx[1:end .!= random_number, 1:end]
    end
    S_reduced = A2S(adj_mtrx)
    SX_reduced_ = S_reduced^k * x_mtrx 
    y_reduced_ = y_mtrx
    return SX_reduced_, y_reduced_
  end 



  ########################### my_function modified ###########################

#adj_matrix = adjacency matrix
  #x_matrix = x matrix
  #y_matrix = y matrix (actual data)
  #paremeter = says which method will be used to split the adj_matrix 
  
  function my_function(adj_matrix, x_matrix, y_matrix, parameter)
    #get Y' data by training the network, before seting the parameter
    k = 3
    S = A2S(adj_matrix)
    SX = S^k * x_matrix
    if parameter == 1
      skiped = round(size(adj_matrix, 1) * 0.25, digits = 0)
      SX_reduced, y_reduced = reduce_size(adj_matrix ,x_matrix, y_matrix, skiped)
      #println("SX_reduced: ", size(SX_reduced))
      #println("y_reduced: ", size(y_reduced'))
      train_x, test_x, train_y, test_y = split_train_test(SX_reduced , y_reduced, 0.7)
      train_x = train_x'
      test_x = test_x'  
      yhot = onehotbatch(vec(train_y), [1, 2]) 
      train_y = yhot
      test_y = onehotbatch(vec(test_y), [1, 2])
      #println("train_x: ", size(train_x))
      #println("train_y: ", size(train_y))
      #println("test_x: ", size(test_x))
      #println("test_y: ", size(test_y))
      model, resDict, epochs, loss = load_and_train_model(train_x, train_y)
      weight = resDict["model"].layers[1].weight
      accuracy = round(mean( onecold( model(test_x), [1, 2] ) .== onecold(test_y, [1, 2]) ) * 100, digits = 2)
      println("Accuracy by training tested data: ", accuracy, "%", ", k = $k")
      #println("weight: ", size(weight))
      #println("SX = ", size(SX))
      #println("weight = ", size(weight))
      #Get the predicted values by using SX matrix and our model
      pred_by_model = onecold(model(SX'), [1, 2])
      #Get the predicted values by using the weight matrix
      SX_x_weight = SX * weight'
      predicted_mtrx = onecold( SX_x_weight' )
      check_accuracy(y_matrix, predicted_mtrx)
      final = SX_reduced * weight'
    elseif parameter == 2
      n_lines_to_split = get_divisors(size(adj_matrix, 2))[2]
      SX_segmented = adj_matrxix_x_X_matrix_dictionary(adj_matrix, x_matrix, Int(n_lines_to_split))
      Ydata_segmented = split_y_matrix(y_matrix, Int(n_lines_to_split))
      combined_data =  combine_data(SX_segmented, Ydata_segmented)
      weight_dict = Dict()
      for i in range(1, length(combined_data))
        train_x, test_x, train_y, test_y = split_train_test(combined_data[i][1] , combined_data[i][2], 0.7)
        train_x = train_x'
        test_x = test_x'  
        yhot = onehotbatch(vec(train_y), [1, 2]) 
        train_y = yhot
        test_y = onehotbatch(vec(test_y), [1, 2])
        #println("train_x: ", size(train_x))
        #println("train_y: ", size(train_y))
        #println("test_x: ", size(test_x))
        #println("test_y: ", size(test_y)) 
        model, resDict, epochs, loss = load_and_train_model(train_x, train_y)
        weight = resDict["model"].layers[1].weight
        accuracy = round(mean( onecold( model(test_x), [1, 2] ) .== onecold(test_y, [1, 2]) ) * 100, digits = 2)
        println("Accuracy by training tested data: ", accuracy, "%", ", k = $k")
        weight_dict["w_$i"] = weight
      end
      weight_sum = zeros(size(weight_dict["w_1"], 1), size(weight_dict["w_1"], 2))
      for i in range(1, length(weight_dict))
        weight_sum +=  weight_dict["w_$i"]
      end 
      weight_average_matrix = weight_sum/length(weight_dict)
      SX_x_weight = SX * weight_average_matrix'
      predicted_mtrx = onecold( SX_x_weight' )
      check_accuracy(y_matrix, predicted_mtrx)
    elseif parameter == 3 
      SX_reduced, y_reduced = radomly_reduced_matrix(adj_matrix ,x_matrix, y_matrix ); 
      train_x, test_x, train_y, test_y = split_train_test(SX_reduced , y_reduced, 0.7);
      train_x = train_x'
      test_x = test_x'  
      yhot = onehotbatch(vec(train_y), [1, 2]) 
      train_y = yhot
      test_y = onehotbatch(vec(test_y), [1, 2])
      model, resDict, epochs, loss = load_and_train_model(train_x, train_y)
      weight = resDict["model"].layers[1].weight
      accuracy = round(mean( onecold( model(test_x), [1, 2] ) .== onecold(test_y, [1, 2]) ) * 100, digits = 2)
      println("Accuracy by training tested data: ", accuracy, "%", ", k = $k")
      final = SX_reduced * weight'

      pred_by_model = onecold(model(SX'), [1, 2])
      #Get the predicted values by using the weight matrix
      SX_x_weight = SX * weight'
      predicted_mtrx = onecold( SX_x_weight' )
      #Check that the two methods give the same number. They should give the same
      check_accuracy(y_matrix, predicted_mtrx)
    end 
    return  accuracy, epochs, loss, predicted_mtrx
  end 

