import redpitaya_scpi as rpscpi

def pollInstanceSample(rps):
    """"
    trigger delay set ot 14192 for 50,60,70,80,90,100
    16192 for 110,120 
    18192 for 130,140,150
    20192 for 160,170,180
    22192 for 190,200
    Increase delay in trigger for max distance coverage #8192
    """
    rps.tx_txt('ACQ:DEC 64')
    rps.tx_txt('ACQ:TRIG:DLY 20192') 
    rps.tx_txt('ACQ:START')
    rps.tx_txt('ACQ:TRIG EXT_PE')
       
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