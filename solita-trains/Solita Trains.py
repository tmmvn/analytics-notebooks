# -*- coding: utf-8 -*-

# -- Sheet --

# # Solita Data Engineer Testin Raportti
# Tämä Jupyter työkirja sisältää Solitan Data Engineer -testin työstöraportin. Tehtävässä oli tarkoituksena ennusta ehtiikö työntekijä ensi viikon torstaina tasan klo 16.15 Tampereella (TPE) tapahtuvaan iltamenoon InterCity IC27:lla (siirtyminen asemalta tapahtumaan kestää tasan 15 minuuttia) vai tulisiko hänen lähteä jo aikaisemmalla junalla?
# 
# Tehtävän vaatimat vaiheet:
# + Hae tarvittu data liikenneviraston avoimen datan rajapinnasta, ja vie se tietokantaan
# + Tee ennuste junan saapumisajasta Tampereen asemalle. Koska tämä ei ole puhtaasti Data Science -harjoitus, ennuste voi olla myös hyvin yksinkertainen (keskiarvo tai vastaava).
# + Esitä perustelut antamaasi ennusteeseen helposti ymmärrettävässä muodossa esim. visualisointien avulla.
# + Luo sovellus joka tarjoaa rajapinnan, joka kertoo, monelta IC27 juna on ensi torstaina Tampereella
# + Mikäli aikaa jää, kirjoita muutamalla lauseella, miten muuten tätä liikenneviraston tarjoamaa avointa dataa voitaisiin hyödyntää liiketoiminnan edistämisessä: mitä mahdollisuuksia datalla olisi, mikä siinä on hyvää ja mitä ongelmia voisi tulla?
#   
# Lisäksi toivottiin läpikäynti tehdyistä työvaiheista sekä työstön aikana tulleita ajatuksista ja mahdollisista ongelmista.
# 
# ## Datan tutkinta
# Aloitin työstön tutkimalla rajapinnan antamaa dataa. Tähän hyödynsin Jupyter Notebookkia, ja alla olevaa koodia.


# API-kyselyihin
import requests
# Tiedon käsittelyyn
import pandas as pd
# Visualisointiin
# Korjaa: Siirrä myöhemmälle, jossa käytetään
from lets_plot import *
from lets_plot.geo_data import *

# Otetaan yhteys rajapintaan. Kokeillaan aluksi vain satunnaista päivämäärää.
# Tässä olisi hyvä tarkistaa, että rajapinta vastaa (koodi 200) ja käsitellä virheet kauniisti
r = requests.get('https://rata.digitraffic.fi/api/v1/trains/2020-05-01/27')
data = r.text
data

# Data näyttää olevan JSON muodossa, joten tuodaan se Pandas DataFrameen
df = pd.read_json(data)
df

# Meitä kiinnostava data näyttää olevan kentässä timeTableRows, joten otetaan siitä uusi dataframe
df_timetable = pd.json_normalize(df.timeTableRows[0])
df_timetable

# Hyödynnetään asematunnusta TPE ja rajataan tiedot saapumismerkintöihin
df_tre = df_timetable[(df_timetable.stationShortCode == "TPE") & (df_timetable.type == "ARRIVAL")]
df_tre

# ## Datan louhinta
# Kun data on nyt pyöritelty, on seuraava vaihe louhia riittävästi dataa analyysin tekemiseen. Tähän on muutamia erilaisia vaihtoehtoja. Päädyin itse hyödyntämään yksinkertaisinta vaihtoehtoa: eli menemään halutusta ajankohdasta taaksepäin 3 kuukautta, poimien jokaisen torstain vuoron tiedot, ja noiden pohjalta muodostamaan arvion keskiarvolla. Tuloksen tarkkuus ei ole kovinkaan luotettava, ja parempaan analyysin olisi hyvä verrata pidempää ajanjaksoa sekä myös muita päiviä erilaisten saapuisaikaan vaikuttavien muuttujien havaitsemiseksi. Kenties tiettyyn vuodenaikaan vuorot ovat enemmän myöhässä, tai kenties torstain vuorot ovat todennäköisemmin myöhässä.
# 
# Myös datan luohinta tietokantaan voidaan suorittaa monella tapaa. Yksi vaihtoehto olisi hakea koko datapaketti ja tallentaa se omaan Data Warehouse -ratkaisuun. Ratkaisun päälle voisi rakentaa ETL-prosessin, jolla tieto saadaan sopiviin Data Martteihin. Tämä olisi hyödyllistä myös ratkaisun laajentamiseksi tarjoamaan useamman junan ja kaupungin tiedot.
# 
# Koska vaatimusta tähän ei kuitenkaan ole, ratkaisutietokanta on yksinkertainen yhden taulun relaatiotietokanta. Valittu teknologia on MariaDB AWS alustalla. Tauluun tallennetaan rajapinnasta päivämäärä, ajastettu aika, sekä toteutuneen ajan erotus. Lisäksi avaimena käytetään juoksevaa automaattisesti kasvavaa indeksiä. Tietokannan toteuttaminen erilaisella schemalla olisi suotavampaa, sillä pelkästä yhden vuoron ja aseman seurannasta toiveena olisi oletettavasti laajentaa tulevaisuudessa kattavampaan palettiin. Tätäkään ei kuitenkaan ole vaadittu, joten mennään taas sieltä, mistä aita on matalin, mutta vaatimukset tulevat täytettyä.
# 
# ### Ongelmia
# Tässä kohtaa törmäsin ongelmaan. Minulla ei ollut valmiina tietokantaa tai sopivia palikoita valmiina kotikoneella, joten ajattelin jatkaa pilvipalveluiden kanssa. Tässä kohtaa törmäsin kuitenkin ongelmaan: Jupyter Notebookin Datalore Kernel ei suostunut asentamaan MariaDB connectoria. Toiseen tietokantaan vaihtaessani törmäsin puolestani ongelmaan, ettei tietokanta vastannut. Tämä oli luultavasti VPC ongelma, jota en ehtinyt tutkia.
# 
# Rupesin tässä kohtaa sitten asentamaan paikallisesti sopivia ratkaisuja, mutta en ehtinyt kovinkaan pitkälle. Java-koodi on lopussa.


# Tuodaan kirjasto helpompaan aikakäsittelyyn
import datetime
# Selvitetään seuraava torstai
today = datetime.date.today()
# Korjaa: Seuraan viikon torstaiksi, nyt voi olla saman viikonkin
next_thursday = today + datetime.timedelta(((3 - today.weekday()) % 7))
# Silmukoidaan ja haetaan tarvittava data
df_full = pd.DataFrame()
# Jostain syystä ensimmäiset tiedot ovat muutaman viikon vanhoja. Tämä voi tarkoittaa
# sitä, että vuoroa ei itseasiassa ajeta. Tämä olisi hyvä huomioida loppuanalyysissä
for weeks in range(0, 15):
    # Tämä checki pitäisi siirtää tuonne ylös huomioimaan oikea next thursday, jolloin
    # silmukka on myös nopeampi.
    if today.weekday() == 4:
        queryDate = next_thursday - datetime.timedelta(7 * weeks + 1)
    else:
        queryDate = next_thursday - datetime.timedelta(7 * weeks)
    queryString = 'https://rata.digitraffic.fi/api/v1/trains/%s/27'%queryDate
    r1 = requests.get(queryString)
    data1 = r1.text
    df1 = pd.read_json(data1)
    df2 = pd.json_normalize(df1.timeTableRows[0])
    df3 = df2[(df2.stationShortCode == "TPE") & (df2.type == "ARRIVAL")]
    df_full = df_full.append(df3)
df_full

# Keskimääräinen junien myöhästyminen minuuteissa
df_average = df_full['differenceInMinutes'].mean()
df_average

df_full['scheduledTime']

# Ylläolevasta tiedosta voidaan nähdä, että 2h eri aikavyöhykkeestä siirtämällä (tai kolme talviajalle), ja 15 minuuttia siirtymään riittää paikalle saapumiseen ajoissa keskimääräisen torstaimyöhästymisen ollessa n. 0.78 minuuttia. Dataa visualisoimalla olisi voinut tehdä mukavan kaavion, josta näkisi historiallisen poikkeaman ja esimerkiksi kerrat, jolloin aika ei olisi riittänyt. Yllä mainitun taistelun sijasta en kuitenkaan ehtinyt visualisointia harmillisesti toteuttaa.
# 
# ## Java-koodi
# Kuten yllä mainittua, törmäsin hienoisiin haasteisiin tietokannan kanssa. Jetbrains Datalore alusta ei suostunut asentamaan MariaDB-kirjastoa ja sitä kautta sopivaa konnektoria. Päädyin sitten asentamaan eri tietokannan AWS RDS palveluun, mutta tuohon konnektroi puolestaan timeouttasi kokoajan. Uskoisin ongelman olleen Amazonin VPC-asetuksissa, mutta en pikaisella vilkaisulla löytänyt ongelmaa, joten päätin käydä toteuttamaan ratkaisua paikallisesti.
# 
# Alla on koodipätkät siitä, mitä ehdin toteuttaa. Eli tietokannan luominen MariaDB Java-konnektorilla. Datamainauksen koodaamisen ehdin juuri ja juuri aloittaa, mutta törmäsin 406 responseen, eli luultavasti jotain headereita en ollut asettanut oikein. En ehtinyt näitä kuitenkaan tutkia.


# import javax.net.ssl.HttpsURLConnection;
# import java.io.BufferedReader;
# import java.io.IOException;
# import java.io.InputStreamReader;
# import java.sql.*;
# import java.util.List;
# import java.util.Map;
# import java.util.Random;
# import java.net.HttpURLConnection;
# import java.net.URL;
# import java.util.Scanner;
# 
# private static void CreateDatabase(Connection connection) {
#         System.out.println("Creating databases");
#         System.out.println("=========================================");
#         System.out.println("Creating tables...");
# 
#         String sqlQuery =
#                 "CREATE OR REPLACE TABLE tpe(id int auto_increment, query_date date, arrival_time datetime, deviation int, primary key(id));\n";
# 
#         try {
#             PreparedStatement statement = connection.prepareStatement(sqlQuery);
#             statement.execute();
# 
#             System.out.println("Finished creating the database. Returning to main menu.\n");
#         } catch (Exception e) {
#             e.printStackTrace();
#             System.out.println("Above error when creating the database. Returning to main menu.\n");
#         }
#     }
# 
#     private static void MineData(Connection connection) {
#         try {
#             URL url = new URL("https://rata.digitraffic.fi/api/v1/trains/2020-05-01/27");
# 
#             HttpsURLConnection conn = (HttpsURLConnection) url.openConnection();
#             conn.setRequestProperty("Accept", "text/plain");
#             conn.setRequestMethod("GET");
#             conn.connect();
# 
#             int responsecode = conn.getResponseCode();
#             if (responsecode != 200) {
#                 Map<String, List<String>> map = conn.getHeaderFields();
#                 System.out.println("Printing All Response Header for URL: " + url.toString() + "\n");
#                 for (Map.Entry<String, List<String>> entry : map.entrySet()) {
#                     System.out.println(entry.getKey() + " : " + entry.getValue());
#                 }
#                 throw new RuntimeException("HTTP connection failed. HttpResponseCode: " + responsecode);
#             } else {
#                 String inline = "";
#                 Scanner scanner = new Scanner(url.openStream());
#                 while (scanner.hasNext()) {
#                     inline += scanner.nextLine();
#                 }
#                 scanner.close();
# 
#                 System.out.println(inline);
#             }
#         }
#         catch (Exception e) {
#             e.printStackTrace();
#             System.out.println("Above error when mining data. Returning to main menu.\n");
#         }
#     }


# ## Korjaukset
# Koska testillä oli aikaraja, palautin harjoituksen hieman vajaavaisena. Päätin kuitenkin yrittää jääräpäisenä uudestaan, sillä datan pyörittely on hauskaa. Yllä olevaan osuuteen on jo tehty korjauksia, ja alta aloitan uudestaan SQL kokeilun.


# Re-testing SQL connection to AWS
# In a real-world solution, using AWS SDK and AWS IAM roles as a different way to access instead of
# hardcoding usernames and such would be desirable. Especially in a notebook.
import sqlalchemy
user = 'username'
password = 'password'
host =  'hostname'
port = '3306' 
database = 'databasename'
engine = sqlalchemy.create_engine('mariadb+mariadbconnector://' + user + ':' + password + '@' + host + ':' + port + '/' + database, echo = True)

