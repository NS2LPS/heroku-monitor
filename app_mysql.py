from bottle import default_app, route, template, static_file, request, response
import time, os, io, base64, json
import MySQLdb
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import subplots
from matplotlib import dates
from pytz import timezone
import json

os.environ['TZ'] = 'Europe/Paris'
time.tzset()
with open('credentials.json','r') as f:
    credentials = json.load(f)

####################
# Data logger database interface
####################
class datalogger:
    def __init__(self, name):
        self.conn = MySQLdb.connect(**credentials)
        self.name = name
        self.query("CREATE TABLE IF NOT EXISTS {0} (time INT, data VARCHAR(512));".format(self.name))
        self.commit()
    def query(self, sql):
        cursor = self.conn.cursor()
        cursor.execute(sql)
        return cursor
    def commit(self):
        self.conn.commit()
    def close(self):
        self.commit()
        self.conn.close()
    def logdata(self, timestamp=None, **data):
        timestamp = int(time.time() ) if timestamp is None else int(timestamp)
        data = json.dumps(data)
        self.query("INSERT INTO {0} VALUES({1},'{2}');".format(self.name, timestamp, data))
        self.commit()
    def query_decode(self, query):
        c = self.query(query)
        d = dict(c.fetchall())
        for k,v in d.items():
            d[k] = json.loads(v)
        return d
    def select_all(self):
        return self.query_decode("SELECT * FROM {0};".format(self.name))
    def select_timespan(self, timespan):
        now = int( time.time() )
        return self.query_decode("SELECT * FROM {0} WHERE time>{1};".format(self.name, now-timespan))
    def reset(self):
        self.query("DROP TABLE {0}".format(self.name))
        self.commit()
        self.query("CREATE TABLE IF NOT EXISTS {0} (time INT, data VARCHAR(512));".format(self.name))
        self.commit()
    def delete_timespan(self, timespan):
        now = int( time.time() )
        self.query("DELETE FROM {0} WHERE time<{1};".format(self.name, now-timespan))


####################
# Helper functions
####################

# Extract one key from the dict returned from the database
def extract(dbdata, key, maxsize=None, convolution=None):
    xy = [ (t, float(data[key])) for t,data in dbdata.items() if key in data]
    if not xy:
        return None
    x,y = list(zip( *sorted( xy ) ))
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if convolution and len(y)>convolution:
        a = np.empty(convolution)
        a[:] = 1./convolution
        y = np.convolve(a, y, 'valid')
        x = x[convolution//2:convolution//2+len(y)]
    if maxsize and len(y)>maxsize :
        l = int( len(y)//(maxsize/2.) )
        y = y[::l]
        x = x[::l]
    return x,y


# Plot data vs time
dateformatter = dates.DateFormatter('%H:%M', tz=timezone('Europe/Paris'))
def plotdate(x,y,ylabel=None):
    fig,ax = subplots()
    ax.plot_date(dates.epoch2num(x), y, 'b-', tz='Europe/Paris')
    ax.xaxis.set_major_formatter(dateformatter)
    fig.autofmt_xdate()
    if ylabel : ax.set_ylabel(ylabel)
    fig.set_figheight(4)
    fig.tight_layout()
    out = io.BytesIO()
    fig.savefig(out, format='png', dpi=400)
    res = base64.b64encode(out.getvalue())
    out.close()
    return res


# Number to string formatting function
def mystr(x, fmt=None):
    if np.isnan(x) : return '??'
    if fmt=='int'  : return str(int(x))
    if fmt=='float': return '{0:.1f}'.format( x )
    if fmt=='date' : return time.asctime( time.localtime( float(x) ) )
    if fmt=='time' :
        ts = time.struct_time(time.localtime(float(x)))
        return '{0:02d}h{1:02d}'.format(ts.tm_hour,ts.tm_min)
    if fmt=='temp' : return '{0:.2f} {1}'.format( *((x,'K') if x>1 else (x*1000,'mK')) )
    return str(x)


#####################
# Figure monitoring
#####################
@route('/<figname>/view')
def main(figname):
    lastupdate = os.path.getmtime(f'./static/{figname}.png')
    elapsed = int(time.time()-lastupdate)/60
    if elapsed <= 1:
        title = 'Last update less than one min ago'
    elif elapsed < 60:
        title = 'Last update {0} min ago'.format(int(elapsed))
    elif elapsed < 1440 :
        h = int(elapsed/60)
        m = int(elapsed%60)
        if m:
            title = 'Last update {0} h {1} min ago'.format(h, m)
        else:
            title = 'Last update {0} h ago'.format(h)
    else:
        title = 'Last update more than one day ago'
    figures=[f'{figname}.png']
    return template('layout_fig',
                    title = title,
                    figures = figures,
                    )

@route('/<figname>/upload', method='POST')
def upload(figname):
    upload = request.files.get('figure')
    upload.save(f'./static/{figname}.png', overwrite=True)
    return 'OK\n'


#####################
# Dilu monitoring
#####################
@route('/<name>')
def main(name):
    # Get timespan
    try:
        timespan = int(request.query.timespan)
    except:
        timespan = request.get_cookie('timespan', secret='mlkjdfgmm223!!g') or 43200
    response.set_cookie('timespan', timespan, secret='mlkjdfgmm223!!g')
    # Retrieve data
    logger = datalogger(name)
    dbdata = logger.select_timespan(timespan)
    logger.close()
    MCRuO2 = extract(dbdata, 'MC RuO2')
    MCCernox = extract(dbdata, 'MC Cernox')
    Still = extract(dbdata, 'Still')
    # List of figures
    figures = []
    # Combine Cernox and RuO2 to get MC temperature
    t1 = MCRuO2[0][0] if MCRuO2 else 100000000000
    t2 = MCCernox[0][0] if MCCernox else 100000000000
    t3 = MCRuO2[0][-1] if MCRuO2 else 0
    t4 = MCCernox[0][-1] if MCCernox else 0
    t_MC = np.arange(min(t1,t2),max(t3,t4),10)
    if len(t_MC):
        T_MCRuO2 = np.interp(t_MC, MCRuO2[0], MCRuO2[1], np.nan, np.nan) if MCRuO2 else np.ones(len(t_MC))*np.nan
        T_MCCernox = np.interp(t_MC, MCCernox[0], MCCernox[1], np.nan, np.nan) if MCCernox else np.ones(len(t_MC))*np.nan
        T_MC = np.where(T_MCRuO2<3, T_MCRuO2, T_MCCernox)
        if np.nanmax(T_MC)>1:
            figures.append( plotdate(t_MC, T_MC, ylabel='MC Temperature (K)') )
        else:
            figures.append( plotdate(t_MC, T_MC*1e3, ylabel='MC Temperature (mK)') )
        if MCRuO2 and MCCernox:
            lastval = MCRuO2[1][-1] if MCRuO2[1][-1]<3 else MCCernox[1][-1]
        elif MCRuO2 and not MCCernox:
            lastval = MCRuO2[1][-1]
        elif not MCRuO2 and MCCernox:
            lastval = MCCernox[1][-1]
        else:
            lastval = np.nan
    else:
        lastval = np.nan
    # Still data
    if Still:
        figures.append( plotdate(Still[0], Still[1], 'Still Temperature (K)') )
    # Last values
    T_MCRuO2 = MCRuO2[1][-1] if MCRuO2 else np.nan
    t_MCRuO2 = MCRuO2[0][-1] if MCRuO2 else np.nan
    T_MCCernox = MCCernox[1][-1] if MCCernox else np.nan
    t_MCCernox = MCCernox[0][-1] if MCCernox else np.nan
    T_Still = Still[1][-1] if Still else np.nan
    t_Still = Still[0][-1] if Still else np.nan
    temperatures = [('MC RuO2', mystr(T_MCRuO2,'temp'),  mystr(t_MCRuO2,'time') ),
                    ('MC Cernox', mystr(T_MCCernox,'temp'),  mystr(t_MCCernox,'time') ),
                    ('Still Cernox', mystr(T_Still,'temp'),  mystr(t_Still,'time') )]
    # Fill template
    return template('layout_dilu',
                    title = 'T = '+ mystr(lastval,'temp'),
                    temperatures = temperatures,
                    figures = figures,
                    )

@route('/<name>/logview')
def logview(name):
    logger = datalogger(name)
    dbdata = logger.select_all()
    logger.close()
    logentries = [ (t,data) for t,data in dbdata.items() ]
    logentries.sort(key = lambda x : -x[0])
    s = 'Log entries for {0} :<BR>\n'.format(name)
    for t,data in logentries:
        s += """{date} : {msg}<BR>\n""".format(date=mystr(t,'date'), msg=str(data) )
    return s

@route('/<name>/log', method='POST')
def log(name):
    logger = datalogger(name)
    logger.delete_timespan(43200)
    logger.logdata(**request.forms)
    logger.close()
    return 'OK\n'

@route('/<name>/reset')
def reset(name):
    logger = datalogger(name)
    logger.reset()
    logger.close()
    return 'OK\n'

@route('/static/<filename>')
def server_static(filename):
    return static_file(filename, root='static')

application = default_app()