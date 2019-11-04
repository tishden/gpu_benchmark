CREATE TABLE client_portfolios (
portfolio_id SMALLINT,
dateIdx SMALLINT,
simulations DOUBLE[])
WITH (FRAGMENT_SIZE = 2000000);

CREATE TABLE ccy1 (
portfolio_id SMALLINT,
dateIdx SMALLINT,
simulations DOUBLE[])
WITH (FRAGMENT_SIZE = 2000000);

CREATE TABLE ccy1_param (
portfolio_id SMALLINT,
dateIdx SMALLINT,
simulations DOUBLE[])
WITH (FRAGMENT_SIZE = 2000000);

CREATE TABLE ccy2 (
portfolio_id SMALLINT,
dateIdx SMALLINT,
simulations DOUBLE[])
WITH (FRAGMENT_SIZE = 2000000);

CREATE TABLE ccy2_param (
portfolio_id SMALLINT,
dateIdx SMALLINT,
simulations DOUBLE[])
WITH (FRAGMENT_SIZE = 2000000);