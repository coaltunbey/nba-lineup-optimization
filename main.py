#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import libraries
import numpy as np
import pandas as pd
from pandas import datetime

# Ignore annoying warning (from sklearn and seaborn)
import warnings 
warnings.filterwarnings('ignore')

from pulp import *

# Limiting floats output to 3 decimal points
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) 

# Define main path
main_path = '/Users/macbookpro/AnacondaProjects/nba'

# Read data
box_score = pd.read_csv(main_path + '/data/BoxScore.csv', sep='|')
players = pd.read_csv(main_path + '/data/Players.csv', sep='|')
teams = pd.read_csv(main_path + '/data/Teams.csv', sep='|')
seasons = pd.read_csv(main_path + '/data/Seasons.csv', sep='|')
fixture = pd.read_csv(main_path + '/data/Fixture.csv', sep='|')
live_fixture = pd.read_csv(main_path + '/data/LiveFixture.csv', sep='|')

# sns.distplot(box_score['Minutes'])

# Convert date column to datetime
fixture['Date'] = pd.to_datetime(fixture['Date'])
box_score['Date'] = pd.to_datetime(box_score['Date'])

# Filter out before 2015 FOR GENERAL
#box_score[box_score['Date'] >= pd.to_datetime('2016-01-01')]
box_score = box_score[(box_score['Date'] >= pd.to_datetime('2016-12-01')) & (box_score['Date'] <= pd.to_datetime('2017-03-01'))]


# Filter out tonights matches
########## Live squad selection

home_performances = box_score[(box_score['Venue'] == 'H') & (box_score['OwnTeamId'].isin(live_fixture['HomeTeamId']))]
away_performances = box_score[(box_score['Venue'] == 'A') & (box_score['OwnTeamId'].isin(live_fixture['AwayTeamId']))]

tonights_matches_box_score = pd.concat([home_performances, away_performances])

box_score = tonights_matches_box_score

## önceki bugünün maçları filtresi
#tonights_matches = pd.concat([live_fixture['HomeTeamId'], live_fixture['AwayTeamId']])
#tonights_matches_box_score = box_score[box_score['OwnTeamId'].isin(tonights_matches)]


# Double counter for double double and triple double
box_score['DoubleCount'] = 0

box_score.loc[box_score['PTS'] > 10, 'DoubleCount'] += 1
box_score.loc[box_score['TOT'] > 10, 'DoubleCount'] += 1
box_score.loc[box_score['A'] > 10, 'DoubleCount'] += 1
box_score.loc[box_score['BL'] > 10, 'DoubleCount'] += 1
box_score.loc[box_score['ST'] > 10, 'DoubleCount'] += 1

# Player scoring 

# Base scoring
box_score['Score'] =  box_score['PTS'] + box_score['3P'] * 0.5 + box_score['TOT'] * 1.25 + box_score['A'] * 1.5 + box_score['ST'] * 2 + box_score['BL'] * 2 - box_score['TO'] * 0.5

# Double bonuses
box_score['Score'] = box_score.apply(lambda x: x['Score'] + 1.5 if x['DoubleCount'] == 2 else x['Score'] + 3 if x['DoubleCount'] >= 3 else x['Score'], axis=1)

# Team selection FOR GENERAL

#main = box_score.merge(players[players['Salary'].notnull()][['PlayerId', 'Salary']], on ='PlayerId', how='left')
main = box_score.merge(players[players['Salary'].notnull()][['PlayerId', 'Salary']], on ='PlayerId', how='left')

main = main[main['Salary'].notnull()]

main = main[['PlayerId', 'Score', 'Salary', 'Date']]

main['Year'] = main['Date'].dt.year

# Average scores for each player
main = main.groupby(['PlayerId', 'Year'])['Score', 'Salary'].mean().reset_index()


###### en iyi çıkan / batan

tot = pd.DataFrame(main.groupby('PlayerId').size(), columns=['Count'])

mul_tot = tot[tot['Count'] > 1]

mul_tot = main[main['PlayerId'].isin(mul_tot.index)]

mul_tot['Shift_PlayerId'] = mul_tot['PlayerId'].shift(1)
mul_tot['Shift_Score'] = mul_tot['Score'].shift(1)

mul_tot = mul_tot[mul_tot['PlayerId'] == mul_tot['Shift_PlayerId']]

mul_tot['Diff'] = (mul_tot['Shift_Score'] - mul_tot['Score']) / mul_tot['Score'] * 100

mul_tot = mul_tot.sort_values(by=['Diff'], ascending=False)

mul_tot = mul_tot.merge(players, on='PlayerId', how='left')[['Player', 'Score', 'Shift_Score' ,'Diff']]

# en iyi çıkan / batan





main = main.sort_values(by=['Score'], ascending=False)

main = main.merge(players[['PlayerId', 'Position']], on='PlayerId', how='left')

# Examine players with multiple possible position
multiples = main[(main['Position'].str.contains('/') == True) | (main['Position'].str.contains('-') == True)]

###### Position Delimiters
multiples['Position'] = multiples['Position'].str.replace('-', ',')
multiples['Position'] = multiples['Position'].str.replace('/', ',')
multiples['Position'] = multiples['Position'].str.split(',')


multiples_dummies = pd.get_dummies(multiples['Position'].apply(pd.Series).stack()).sum(level=0)

multiples = pd.concat([multiples, multiples_dummies], axis=1)

del multiples['Position']

multiples

def multiple_position_handler(x):
    if 'F' in x:
        if(x['F'] == 1):
            x['PF'] = 1
            x['SF'] = 1

    if 'G' in x:
        if(x['G'] == 1):
            x['SG'] = 1
            x['PG'] = 1
        
    return x

multiples = multiples.apply(lambda x : multiple_position_handler(x), axis=1)

if 'F' in multiples:
    del multiples['F']

if 'G' in multiples:
    del multiples['G']

not_multiples = main[(main['Position'].str.contains('/') == False) & (main['Position'].str.contains('-') == False)]

not_multiples = pd.get_dummies(not_multiples)

def not_multiple_position_handler(x):
    if 'Position_F' in x:
        if(x['Position_F'] == 1):
            x['Position_PF'] = 1
            x['Position_SF'] = 1
            
    if 'Position_G' in x:
        if(x['Position_G'] == 1):
            x['Position_SG'] = 1
            x['Position_PG'] = 1
        
    return x

not_multiples = not_multiples.apply(lambda x : not_multiple_position_handler(x), axis=1)


if 'Position_F' in not_multiples:
    del not_multiples['Position_F']

if 'Position_G' in not_multiples:
    del not_multiples['Position_G']


# Column renaming
not_multiples.columns = multiples.columns.tolist()

main = pd.concat([multiples, not_multiples])

main = main.drop(main[main['PlayerId'] == 754].index)
main = main.drop(main[main['PlayerId'] == 189].index)

########### Linear programming

player_ids = main['PlayerId'].astype(str)
player_salaries = main['Salary']
player_scores = main['Score']
player_c = main['C']
player_pf = main['PF']
player_pg = main['PG']
player_sf = main['SF']
player_sg = main['SG']


player_salariesx = dict(zip(player_ids, player_salaries))
player_scoresx = dict(zip(player_ids, player_scores))

player_cx = dict(zip(player_ids, player_c))
player_pfx = dict(zip(player_ids, player_pf))
player_pgx = dict(zip(player_ids, player_pg))
player_sfx = dict(zip(player_ids, player_sf))
player_sgx = dict(zip(player_ids, player_sg))

player_ids = main['PlayerId'].astype(str).tolist()

W = 50000
maxplayer = 8
minplayer = 5

x = LpVariable.dicts('PlayerId', player_ids, 0, 1, LpBinary)

prob=LpProblem('knapsack', LpMaximize)

# Objective Function
cost = lpSum([ player_scoresx[i]*x[i] for i in player_ids])
prob += cost

# Constraints
prob += lpSum([player_salariesx[i]*x[i] for i in player_ids]) <= W

prob += lpSum([x[i] for i in player_ids]) <= maxplayer
prob += lpSum([x[i] for i in player_ids]) >= minplayer

prob += lpSum([player_cx[i]*x[i] for i in player_ids]) >= 1
prob += lpSum([player_pfx[i]*x[i] for i in player_ids]) >= 1
prob += lpSum([player_pgx[i]*x[i] for i in player_ids]) >= 1
prob += lpSum([player_sfx[i]*x[i] for i in player_ids]) >= 1
prob += lpSum([player_sgx[i]*x[i] for i in player_ids]) >= 1

prob += lpSum([player_sgx[i]*x[i] + player_pgx[i]*x[i] for i in player_ids]) == 4
prob += lpSum([player_sfx[i]*x[i] + player_pfx[i]*x[i] for i in player_ids]) == 4

# Solve
prob.solve()
print(LpStatus[prob.status])

# Get results
result = {}

for i in player_ids: 
    print(i, value(x[i]))
    result[float(i)] = value(x[i])
        
squad = []

for i,k in result.items():
    if k == 1:
        squad.append(i)

# Take a look at the selected players
players[players['PlayerId'].isin(squad)]
main[main['PlayerId'].isin(squad)]

# Total value obtained    
print(value(prob.objective))

# Total salary spent
print(sum([ player_salariesx[i]*value(x[i]) for i in player_ids]))


current_teams = box_score[box_score['PlayerId'].isin(squad)][['PlayerId', 'OwnTeamId']].drop_duplicates()

current_teams.merge(teams, left_on='OwnTeamId', right_on='TeamId', how='left')




