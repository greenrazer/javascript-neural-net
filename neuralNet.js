$(document).ready(function() {		
	createNeuralNet(2, 3, 2);

	$("#continueNet").click(function() {
		 for(let i = 0; i<499; i++){
			runNeuralNet();
		 }
		runNeuralNet(true);
	});
});