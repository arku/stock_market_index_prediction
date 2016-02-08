require 'csv'
CSV.foreach('nifty50.csv', headers:true) do |row|
    print row
    puts "\n"
end
