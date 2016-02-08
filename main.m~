
	%Load data matrix

	fprintf('Reading nifty data...\n');
	data_read = csvread('nifty50.csv');
	fprintf(strcat("Read", num2str(size(data_read, 1)), " rows"));

	% Skip the date column, volume and change%
	% Skip the first row since it contains the column headers

	fprintf('\nPreprocessing...\n');
	data = data_read(2:size(data_read, 1), 3:6);
	size(data)
	
	% Compute the feature vector from the data

	featureVector = computeFeatures(data, 10);
	sz = size(featureVector)


