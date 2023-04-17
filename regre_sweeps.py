for data in ['boston', 'concrete', 'diabetes']:
    coms = ''
    for lr in [5e-4, 1e-3, 5e-3, 1e-2]:
        for num_hidden in [16, 32, 64]:
            coms += "python uci_mnf.py --regress 1 --lr_0 %f --data_name %s --num_hidden %i\n" % (lr, data, num_hidden)
                
    with open('%s_sweeps.sh' % (data), 'w') as f:
        f.write(coms)