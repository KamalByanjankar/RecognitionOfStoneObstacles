import redpitaya_scpi as rpscpi

def pollInstanceSample(rps):
    rps.tx_txt('ACQ:DEC 64')
    rps.tx_txt('ACQ:TRIG EXT_PE')
    rps.tx_txt('ACQ:TRIG:DLY 8192')
#     rps.tx_txt('ACQ:TRIG:LEVEL 100')
    rps.tx_txt('ACQ:START')
       
    while 1:
        rps.tx_txt('ACQ:TRIG:STAT?')
        if rps.rx_txt() == 'TD':
            break
        
    rps.tx_txt('ACQ:SOUR1:DATA?')
    
    buff_string = rps.rx_txt()
    buff_string = buff_string.strip('{}\n\r').replace("  ", "").split(',')
    buff = list(map(float, buff_string))
    
    return buff

# def runner(ip='192.168.0.156'):
def runner(ip='192.168.128.1'):
    rps = rpscpi.scpi(ip)
    print('Connected to Device, Acquiring Data Now...')
    return rps