$(document).ready(function() {	
	var arr1 = [[1,2],[3,4]];
	var arr2 = [[4,3],[2,1]];
	
	var matrix1 = matrixMultiply(arr1,arr2);
	
	createNeuralNet(2, 3, 2);
	
	var keepGoing = false;
	var doit;
	
	$("#continueNet").click(function() {
		for(let i = 0; i<499; i++){
			runNeuralNet();
		}
		runNeuralNet(true);
	});
});