$(document).ready(function() {
	let trainin = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]];
	let trainout = [[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1],[0,0,0]];


	setTrainingData(trainin,trainout);		
	createNeuralNet(3, 4, 4, 3);


	$("#continueNet").click(function() {
		 for(let i = 0; i<19999; i++){
			runNeuralNet();
		 }
		runNeuralNet(true);
	});
});