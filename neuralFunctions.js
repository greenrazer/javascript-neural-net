//default Matrix
var LAYERS = [2, 3, 2];
var TRAINING_IN = [[0,0],[0,1],[1,0],[1,1]];
var TRAINING_OUT = [[0,1],[1,0],[1,1],[0,0]];

var matrixId = 0;
var eConstant = 2.7182818284;
var weightMatrixes;


/* sets the training data for the neural net
* @param (array) train_in - the training data for the input
* @param (array) train_out - the training data for the output
*/
function setTrainingData(train_in, train_out){
	TRAINING_IN = scaleValues(train_in);
	TRAINING_OUT = scaleValues(train_out)

}


/*
* number of integers to describe neural net
* @param (integer(s)) **args - any number of integers to describe neural net(e.g. 2,3,1)
*/
function createNeuralNet(){

	//define the network
	if(arguments.length > 2){
		LAYERS = [];
		for(i = 0; i < arguments.length; i++){
    	    LAYERS.push(arguments[i]); 
    	}
    }
    if(LAYERS[0] !== TRAINING_IN[0].length)
    	console.log("your input layer doesnt match the amount of inputs on your training input array");
    if(LAYERS.last() !== TRAINING_OUT[0].length)
    	console.log("your output layer doesnt match the amount of inputs on your training output array");

    //create the weight matrixes
    weightMatrixes = [];
    for(let i = 0; i<LAYERS.length-1; i++){
    	weightMatrixes.push(createArray(LAYERS[i],LAYERS[i+1]));
    }
    weightMatrixes = fillWeightMatrixes(weightMatrixes);
    
    draw2DArray(TRAINING_IN);
}

function runNeuralNet(printArray){
	printArray = printArray || false;
	//forward propagate the network
	let calcHistory = forwardProp(weightMatrixes);

    let cost = computeCost(TRAINING_OUT, calcHistory.last());
    
    //print the cost
    //console.log(cost);
    
    //draw yHat to screen
    if(printArray){
    	draw2DArray(calcHistory.last());
		draw2DArray(calcHistory.last(),true);
	}

    //use back propagation to find the weight change
    let approxWeightChange = computeNumericalGradient(weightMatrixes);
    let weightChange = backProp(calcHistory, weightMatrixes);
    let unrolledWeightChange = unroll($.extend(true, [], weightChange).reverse());
    console.log(unrolledWeightChange, approxWeightChange);
    console.log(vectorSubtract(unrolledWeightChange, approxWeightChange));

    //change the weights
    weightMatrixes = changeWeights(weightMatrixes, weightChange, 3);
}

/*
* Fills the weight matrices with random values
* @param (array) arr = array of weight matricies
* @return (array) arr = arrays filled with weights
*/
function fillWeightMatrixes(arr){
	for(let i = 0;i<arr.length;i++){
		let height = arr[i].length;
		let width = arr[i][0].length;
		for(let j = 0; j<height; j++){
			for(let q = 0; q<width; q++){
				let tempRange = Math.sqrt(6/(LAYERS[i]+LAYERS[i+1]));
				arr[i][j][q]=Math.random(-tempRange,tempRange);
			}
		}
		//draw2DArray(arr[i]);
	}
	return arr;
}

/*
* add a constant to each element of a matrix
* @param (array) arr - array to add numbers to
* @param (float) num - number to add
* @return (array) arr - the added array
*/
function matrixAddConstant(arr, num){
	let height = arr[i].length;
	let width = arr[i][0].length;
	for(let i = 0; i<height; i++){
		for(let j = 0; j<width; j++){
			arr[i][j]=arr[i][j]+num;
		}
	}
	//draw2DArray(arr[i]);
	return arr;
}

/*
* change the weights by subtracting the change in weights times a scalar
* @param (array) weights - weights array
* @param (array) dxWeights - the changes in weights
* @param (float) scalar - number to multiply the dxWeights by in order to make meaningful change
* @return (array) weights - the updated weights array
*/
function changeWeights(weights, dxWeights, scalar){
	for(let i = 0; i<weights.length; i++){
		weights[i]=matrixSubtract(weights[i], matrixMultiplyConstant(dxWeights[dxWeights.length-(i+1)],scalar));
	}
	return weights;
}

/*
* computes the error between two matrixies
* @param (array) arr1 - first array to be compared
* @param (array) arr2 - second array to be compared
* @return (float) error - the error between the first and second array
*/
function computeCost(arr1, arr2){
	let height1 = arr1.length;
	let width1 = arr1[0].length;
	let height2 = arr2.length;
	let width2 = arr2[0].length;	
// 	draw2DArray(arr1);
// 	draw2DArray(arr2);
	
	if(width1 !== width2 || height1 !== height2){ //make sure you can actually multiply the matrices correctly
		console.log("matrix sizes are not compatible for subtraction");
		return null;
	}

	let sum = 0;
	let error = 0;
	for(let i = 0; i<height1;i++){
		for(let j = 0; j<width1; j++){
			sum += Math.pow(arr1[i][j]-arr2[i][j],2);
		}
		
		error += sum/2;
	}
	return error;
}

function computeCostTEST(arr1, arr2){
	let height1 = arr1.length;
	let width1 = arr1[0].length;
	let height2 = arr2.length;
	let width2 = arr2[0].length;	
// 	draw2DArray(arr1);
// 	draw2DArray(arr2);
	
	if(width1 !== width2 || height1 !== height2){ //make sure you can actually multiply the matrices correctly
		console.log("matrix sizes are not compatible for subtraction");
		return null;
	}

	let sum = 0;
	let error = 0;
	let errorArr = matrixSubtract(arr1, arr2);
	for(let i = 0; i<height1;i++){
		for(let j = 0; j<width1; j++){
			sum += Math.pow(errorArr[i][j],2);
		}
	}
	sum = sum/2;
	return sum;
}

/*
* forward propagates the neural network to get yHat
* @param (array) arr - weights array;
* @return (array) history - all the matrixes that led up to getting and yHat
*/
function forwardProp(weights, in_arr){
	let startMatrix = in_arr || TRAINING_IN
	let tempMatrix;
	let history = [];
	history.push(tempMatrix = matrixMultiply(startMatrix,weights[0]));
	history.push(tempMatrix = sigmoid(tempMatrix));
	for(let i = 1;i<weights.length;i++){
		history.push(tempMatrix=matrixMultiply(tempMatrix,weights[i]));
		history.push(tempMatrix=sigmoid(tempMatrix));
	}
	return history;
}

/*
* back propagates the neural network to get the change in weights
* @param (array) history - all the matrixes that led up to getting yHat
* @param (array) weights - actual weight array
* @return (array) dJdW - derivative of dJ for each dW
*/
function backProp(history, weights){
	history.unshift(TRAINING_IN);
	let dJdW = [];
	let backPropError = matrixMultiplyElements(matrixMultiplyConstant(matrixSubtract($.extend(true, [], TRAINING_OUT),history.pop()),-1),dxSigmoid(history.pop()));
	dJdW.push(matrixMultiply(transposeMatrix(history.pop()),backPropError));
	
	let count = 0;
	while(history.length > 0){
		backPropError = matrixMultiplyElements(matrixMultiply(backPropError,transposeMatrix(weights[weights.length-(count+1)])),dxSigmoid(history.pop()));
		dJdW.push(matrixMultiply(transposeMatrix(history.pop()),backPropError));
		count++;
	}
	return dJdW;
}

function computeNumericalGradient(weights){
	let tempWeightsPlus;
	let tempWeightsMinus;
	// console.log(weights.length, weights[0].length, weights[0][0].length)
	// console.log(tempWeightsPlus.length, tempWeightsPlus[0].length, tempWeightsPlus[0][0].length)
	// console.log(tempWeightsMinus.length, tempWeightsMinus[0].length, tempWeightsMinus[0][0].length)
	// console.log(weights, tempWeightsPlus, tempWeightsMinus)
	let tempLossPlus;
	let tempLossMinus;
	let approxdJdW = [];
	let epsilon = 0.0001;
	for(let i = 0; i<weights.length; i++){
		for(let j = 0; j<weights[i].length; j++){
			tempWeightsPlus = $.extend(true, [], weights);
			tempWeightsMinus = $.extend(true, [], weights);
			for(let q = 0; q<weights[i][j].length; q++){
				tempWeightsPlus[i][j][q] += epsilon;
				tempWeightsMinus[i][j][q] -= epsilon;

				tempLossPlus = computeCostTEST(TRAINING_OUT, forwardProp(tempWeightsPlus,TRAINING_IN).last());
				tempLossMinus = computeCostTEST(TRAINING_OUT, forwardProp(tempWeightsMinus,TRAINING_IN).last());
				approxdJdW.push((tempLossPlus-tempLossMinus)/(2*epsilon));
			}
		}
	}
	return approxdJdW;


}

function unroll(arr) {
	let finalArr = [];
	for(let i = 0; i<arr.length; i++){
		for(let j = 0; j<arr[i].length; j++){
			for(let q = 0; q<arr[i][j].length; q++){
				finalArr.push(arr[i][j][q]);
			}
		}
	}
	return finalArr

}

function copy(o) {
   var output, v, key;
   output = Array.isArray(o) ? [] : {};
   for (key in o) {
       v = o[key];
       output[key] = (typeof v === "object") ? copy(v) : v;
   }
   return output;
}

/*
* makes an array of variable size (e.g. createArray(2, 2, 2) = [[[],[]],[[],[]]])
* @param (integer(s)) length - size of each demention of the array
* @return (array) arr - array of size specified
*/
function createArray(length) {
   var arr = new Array(length || 0),
        i = length;

   if (arguments.length > 1) {
        var args = Array.prototype.slice.call(arguments, 1);
        while(i--) arr[length-1 - i] = createArray.apply(this, args);
    }

    return arr;
}
	
/*
* multiplies 2 arrays together matrix style
* @param (array) arr1 - first matrix to be multiplied
* @param (array) arr2 - second matrix to be multiplied
* @return (array) finalArr - multiplied together matrix
*/
function matrixMultiply(arr1, arr2){
	let height1 = arr1.length;
	let width1 = arr1[0].length;
	let height2 = arr2.length;
	let width2 = arr2[0].length;
	
	//console.log(arr1);
	//console.log(arr2);
	//console.log(height1+"x"+width1+"*"+height2+"x"+width2);
	
	if(width1 !== height2){ //make sure you can actually multiply the matrices correctly
		console.log("matrix sizes are not compatible for multiplication");
		return null;
	}

	let finalArr = createArray(height1, width2);
	let sum;
	for(let i = 0; i<height1;i++){
		for(let q = 0; q<width2; q++){
			sum = 0; //set/reset sum to zero
			for(let j = 0; j<width1; j++){
				sum += arr1[i][j]*arr2[j][q];//multiply the ij element from array one and jq element from array 2 and add them the sum
				//console.log("arr1["+i +"]["+j +"]("+arr1[i][j]+")*arr2["+ j+"]["+ q +"]("+arr2[j][q]+") = "+ sum);
			}
			finalArr[i][q] = sum;
			//console.log(finalArr[i][q]);
		}
	}
	return finalArr;
}

/*
* multiplies 2 matrixes by element
* @param (array) arr1 - first matrix to be multiplied
* @param (array) arr2 - second matrix to be multiplied
* @return (array) arr1 - product matrix
*/
function matrixMultiplyElements(arr1, arr2){
	let height1 = arr1.length;
	let width1 = arr1[0].length;
	let height2 = arr2.length;
	let width2 = arr2[0].length;
	
	if(width1 !== width2 || height1 !== height2){ //make sure you can actually multiply the matrices correctly
		console.log("matrix sizes are not compatible for element wise multiplication");
		return null;
	}

	for(let i = 0; i<height1;i++){
		for(let j = 0; j<width1; j++){
			arr1[i][j]=arr1[i][j]*arr2[i][j];
		}
	}
	return arr1;
}

/*
* multiplies each elelment in a matrix by a constant
* @param (array) arr - matrix to be multiplied
* @param (float) num - number to multiply each element in the matrix by
* @return (array) arr - product matrix
*/
function matrixMultiplyConstant(arr, num){
	let height = arr.length;
	let width = arr[0].length;

	for(let i = 0; i<height;i++){
		for(let j = 0; j<width; j++){
			arr[i][j]=arr[i][j]*num;
		}
	}
	return arr;
}

/*
* scales All elements in a matrix by the highest number in the matrix, or a value if defined 
* @param (array) arr - matrix to be scaled
* @param (float) val - number to scale matrix by
* @return (array) arr - product matrix
*/
function scaleValues(arr, val){
	let tempArr = arr;
	let height = arr.length;
	let width = arr[0].length;

	if(val){
		tempArr = matrixMultiplyConstant(tempArr, (1/val));
	}
	else{
		tempArr = matrixMultiplyConstant(tempArr, (1/findMax(tempArr)));
	}

	return tempArr;
}

/*
* finds the maximum element in a matrix
* @param (array) arr - matrix to find max in
* @return (Integer) currMax - maximum Element
*/
function findMax(arr){
	let height = arr.length;
	let width = arr[0].length;
	let currMax = arr[0][0];
	for(let i = 0; i<height;i++){
		for(let j = 0; j<width; j++){
			if(arr[i][j] > currMax){
				currMax = arr[i][j];
			}
		}
	}
	return currMax;
}

/*
* subtracts 2 matrixes
* @param (array) arr1 - first matrix
* @param (array) arr2 - second matrix to be subtracted from first
* @return (array) arr1 - sum of matrixes
*/
function matrixSubtract(arr1, arr2){
	let height1 = arr1.length;
	let width1 = arr1[0].length;
	let height2 = arr2.length;
	let width2 = arr2[0].length;
	let finalArr = createArray(height1, width1);
	
	if(width1 !== width2 || height1 !== height2){ //make sure you can actually multiply the matrices correctly
		console.log("matrix sizes are not compatible for subtraction");
		return null;
	}

	for(let i = 0; i<height1;i++){
		for(let j = 0; j<width1; j++){
			finalArr[i][j] = arr1[i][j]-arr2[i][j];
		}
	}
	return finalArr;
}

function vectorSubtract(arr1, arr2){
	let length1 = arr1.length;
	let length2 = arr2.length;
	let finalVect = createArray(length1);
	
	if(length1 !== length2){ //make sure you can actually multiply the matrices correctly
		console.log("matrix sizes are not compatible for subtraction");
		return null;
	}

	for(let i = 0; i<length1;i++){
		finalVect[i] = arr1[i]-arr2[i];
	}
	return finalVect;
}

/*
* transposes a matrix
* @param (array) arr - matrix in which to be transposed
* @return (array) finalArr - transposed matrix
*/
function transposeMatrix(arr){
	let height = arr.length;
	let width = arr[0].length;
	
	let finalArr = createArray(width, height);
	for(let i = 0; i <width; i++){
		for(let j = 0; j <height; j++){
			finalArr[i][j] = arr[j][i];
		}
	}
	return finalArr;
}

/*
* draws a two dimentional array on the screen
* @param (array) arr - 2 dimentional array to draw on screen
* @param (boolean) round - whether or not to round the array values to nearest integer
*/
function draw2DArray(arr, round){
	let height = arr.length;
	let width = arr[0].length;
	let tempArr = (round)?roundMatrix(arr):arr;
	
	//console.log(arr);
	
	$("body").append("<div id=\"matrix"+ matrixId+ "\" class=\"matrix\"></div>");
	$("#matrix"+matrixId).append("<ul style=\"list-style-type: none;\"></ul>");
	for(let i = 0; i < height; i++){
		$("#matrix"+matrixId+" ul").append("<li>|"+tempArr[i]+"|</li>");
	}
	matrixId++;
}

/*
* rounds every element of a matrix to the nearest Integer
* @param (array) arr - matrix to round each element
* @return (array) roundedArr - matrix that has been rounded 
*/
function roundMatrix(arr){
	let height = arr.length;
	let width = arr[0].length;
	roundedArr = $.extend(true, [], arr);

	for(let i = 0; i <height; i++){
		for(let j = 0; j <width; j++){
			roundedArr[i][j] = Math.round(roundedArr[i][j]); 
		}
	}
	return roundedArr

}

/*
* runs sigmoid function on every element of a matrix
* @param (array) arr - matrix to run each element through a sigmoid function
* @return (array) finalArr - matrix that has been sigmoided 
*/
function sigmoid(arr){
	let height = arr.length;
	let width = arr[0].length;
	let finalArr = createArray(height, width);
	
	
	for(let i = 0; i <height; i++){
		for(let j = 0; j <width; j++){
			finalArr[i][j] = 1/(1+Math.pow(eConstant, (-1*arr[i][j]))); 
		}
	}
	return finalArr;

}

/*
* runs derivative of sigmoid function on every element of a matrix
* @param (array) arr - matrix to run each element through a differential sigmoid function
* @return (array) finalArr - matrix that has been derivative sigmoided 
*/
function dxSigmoid(arr){
	let height = arr.length;
	let width = arr[0].length;
	let finalArr = createArray(height, width);
	
	
	for(let i = 0; i <height; i++){
		for(let j = 0; j <width; j++){
			finalArr[i][j] = Math.pow(eConstant, arr[i][j])/Math.pow((1+Math.pow(eConstant, arr[i][j])),2);
		}
	}
	return finalArr;

}

Array.prototype.last = function() {
    return this[this.length-1];
}

