function featureVector = computeFeatures(data, days)



  closing_prices = data(:,1);
  high_prices = data(:,3);
  low_prices = data(:,4);

  sz = size(closing_prices);

  %  ***********************************************
  %   1. Code for computing Simple Moving Average
  %
  %
  %Find the simple moving average from the closing prices
  
  simple_mov_avg = zeros(size(closing_prices));

  % Skip the last 9 rows because we can't compute simple moving average
  for i=1:size(simple_mov_avg,1)-9
	simple_mov_avg(i) = mean(closing_prices(i:i+ days - 1));
  end
  %fprintf("Size of simple moving average is:")
  size(simple_mov_avg);

  %  ***********************************************
  %   2. Code for computing Exponential Moving Average
  %
  %

  %Slicing closing_prices column vector from the data matrix
  simple_avg = mean(closing_prices(1:days));

  % alpha is a smoothing factor
  alpha = 2/days;

  % slice the closing price from 10th day to the end

  X = closing_prices(days:end);
  X(1) = simple_avg;

  exp_mov_avg = filter(alpha, [1 alpha-1], X, simple_avg*(1-alpha));

  % Fill the last 9 values with NaN because they are undefined

  exp_mov_avg = [exp_mov_avg;NaN(days-1,1)];
  %fprintf("Size of exponential moving average is:");

  size(exp_mov_avg);
  featureVector = [simple_mov_avg, exp_mov_avg];

  %  ***********************************************
  %   3. Code for computing Momentum
  %
  %

  %for i=1:size(closing_prices,1)-9
	  %momentum(i) = closing_prices(i) - closing_prices(i+9);
  %end
   
  %Vectorized implementation
  momentum = closing_prices(1:size(closing_prices,1)-9,1) - closing_prices(days:end);
  %momentum(1:3)
  % Pad zeroes at the end
  momentum = [momentum;zeros(days-1,1)];

  % Add momentum to the feature vector
  featureVector = [simple_mov_avg, exp_mov_avg, momentum];


  %  ***********************************************
  %   4. Code for computing stochastic K%
  %
  %

  stochastic_k = zeros(sz);
  for i=1:sz-9
	minimum = min(low_prices(i:i+9,1));
	stochastic_k(i) = [( closing_prices(i) - minimum ) / ( max(high_prices(i:i+9,1)) - minimum )] * 100;
  end
  %stochastic_k(1)

  % Add stochastic_k to the feature vector
  featureVector = [simple_mov_avg, exp_mov_avg, momentum, stochastic_k];
  size(featureVector);

  %  ***********************************************
  %   5. Code for computing stochastic D%
  %
  %
  
  stochastic_d = zeros(sz);
  for i = 1:sz-9
	stochastic_d(i) = mean(stochastic_k(i:i+days-1,1));
  end
  %mean(stochastic_k(12:21,1)) == stochastic_d(12)

  % Add stochastic_d to the feature vector
  featureVector = [simple_mov_avg, exp_mov_avg, momentum, stochastic_k, stochastic_d];
  
  %  ***********************************************
  %   6. Code for computing Commodity Channel Index
  %
  %

  	mean_price = (closing_prices + high_prices + low_prices) / 3;
        size(mean_price)
	
        avg_mean_price = zeros(sz);
	for i = 1:sz-9
		 avg_mean_price(i) = mean(mean_price(i:i+days-1,1));
	end
        size(avg_mean_price)

        mean_deviation = zeros(sz);
        for i = 1:sz-9
		 mean_deviation(i) = mean(abs(mean_price(i:i+9,1) - avg_mean_price(i) * ones(days,1)) );
	end
	size(mean_deviation)
	
	% Problem with vector addition
	
	%commodity_channel_index = zeros(sz);
	%for i=1:sz-9	
        % Vectorized implementation
	        commodity_channel_index =  ( mean_price - avg_mean_price ) ./ (0.015 * mean_deviation);
	%end
	commodity_channel_index(1:10,1)
        size(commodity_channel_index)

	% Add commodity_channel_index to the feature vector
	featureVector = [simple_mov_avg, exp_mov_avg, momentum, stochastic_k, stochastic_d, commodity_channel_index];

  %  ***********************************************
  %   7. Code for Computing accumulation/distribution oscillator
  %
  %
  acc_dis_oscillator = zeros(sz)
  acc_dis_oscillator = high_prices(1:sz(1)-1,1) - closing_prices(2:end,1) ./ ( high_prices(1:sz(1)-1,1) - low_prices(1:sz(1)-1,1) );
  size(acc_dis_oscillator)
  acc_dis_oscillator
end
