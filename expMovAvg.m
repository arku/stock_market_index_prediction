function exponential_moving_average = expMovAvg(data, days)

  closing_prices = data(:,1);

  %Slicing closing_prices column vector from the data matrix
  simple_avg = mean(closing_prices(1:days));

  % alpha is a smoothing factor
  alpha = 2/days;

  % slice the closing price from 10th day to the end

  X = closing_prices(days:end);
  X(1) = simple_avg;

  exponential_moving_average = filter(alpha, [1 alpha-1], X, simple_avg*(1-alpha));

  % Fill the last 9 values with NaN because they are undefined

  exponential_moving_average = [exponential_moving_average;NaN(days-1,1)];

end
