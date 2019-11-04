CREATE TABLE airline
(
YearOfFlight Int32,
Quarter Int32,
MonthOfFlight Int32,
DayofMonth Int32,
DayOfWeek Int32,
FlightDate Date,
UniqueCarrier String,
AirlineID String,
Carrier String,
TailNum String,
FlightNum Int32,
OriginAirportID Int32,
OriginAirportSeqID Int32,
OriginCityMarketID Int32,
Origin String,
OriginCityName String,
OriginState String,
OriginStateFips Int32,
OriginStateName String,
OriginWac Int32,
DestAirportID Int32,
DestAirportSeqID Int32,
DestCityMarketID Int32,
Dest String,
DestCityName String,
DestState String,
DestStateFips Int32,
DestStateName String,
DestWac Int32,
CRSDepTime Int32,
DepTime Int32,
DepDelay Float32,
DepDelayMinutes Float32,
DepDel15 Float32,
DepartureDelayGroups Int32,
DepTimeBlk String,
TaxiOut Float32,
WheelsOff Int32,
WheelsOn Int32,
TaxiIn Float32,
CRSArrTime Int32,
ArrTime Int32,
ArrDelay Float32,
ArrDelayMinutes Float32,
ArrDel15 Float32,
ArrivalDelayGroups String,
ArrTimeBlk String,
Cancelled Float32,
CancellationCode String,
Diverted Float32,
CRSElapsedTime Float32,
ActualElapsedTime Float32,
AirTime Float32,
Flights Float32,
Distance Float32,
DistanceGroup String,
CarrierDelay String,
WeatherDelay String,
NASDelay String,
SecurityDelay String,
LateAircraftDelay String,
FirstDepTime String,
TotalAddGTime String,
LongestAddGTime String,
DivAirportLandings String,
DivReachedDest String,
DivActualElapsedTime String,
DivArrDelay String,
DivDistance String,
Div1Airport String,
Div1AirportID String,
Div1AirportSeqID String,
Div1WheelsOn String,
Div1TotalGTime String,
Div1LongestGTime String,
Div1WheelsOff String,
Div1TailNum String,
Div2Airport String,
Div2AirportID String,
Div2AirportSeqID String,
Div2WheelsOn String,
Div2TotalGTime String,
Div2LongestGTime String,
Div2WheelsOff String,
Div2TailNum String,
Div3Airport String,
Div3AirportID String,
Div3AirportSeqID String,
Div3WheelsOn String,
Div3TotalGTime String,
Div3LongestGTime String,
Div3WheelsOff String,
Div3TailNum String,
Div4Airport String,
Div4AirportID String,
Div4AirportSeqID String,
Div4WheelsOn String,
Div4TotalGTime String,
Div4LongestGTime String,
Div4WheelsOff String,
Div4TailNum String,
Div5Airport String,
Div5AirportID String,
Div5AirportSeqID String,
Div5WheelsOn String,
Div5TotalGTime String,
Div5LongestGTime String,
Div5WheelsOff String,
Div5TailNum String,
Div6TailNum String
)
ENGINE = MergeTree(FlightDate, (YearOfFlight, FlightDate), 8192);