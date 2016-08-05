$(document).ready(function() {	
	var arr1 = [[1,2],[3,4]];
	var arr2 = [[4,3],[2,1]];
	
	var matrix1 = matrixMultiply(arr1,arr2);
	
	createNeuralNet(2, 3, 2);
	
	$("#continueNet").click(function() {
		runNeuralNet();
	});
});