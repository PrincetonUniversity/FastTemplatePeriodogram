

def rms_resid_over_rms(cn, sn, Tt, Yt):
    # This is fairly slow; is there a better way to get best fit pars?
    a, b, c = LMfit(Tt, Yt, np.ones(len(Tt))*0.0001, cn, sn, 2 * np.pi, sgn=True)
    Ym = ftp.fitfunc(Tt, 1, 2 * np.pi, cn, sn, a, b, c)

    S = sqrt(np.mean(np.power(Yt, 2)))

    Rp = sqrt(np.mean(np.power(Ym - Yt, 2))) / S

    a, b, c = LMfit(Tt, Yt, np.ones(len(Tt))*0.0001, cn, sn, 2 * np.pi, sgn=False)
    Ym = ftp.fitfunc(Tt, -1, 2 * np.pi, cn, sn, a, b, c)

    Rn = sqrt(np.mean(np.power(Ym - Yt, 2))) / S
    return min([ Rn, Rp ])

rms = lambda x : sqrt(np.mean(np.power(x, 2)))


def match_up_truncated_template(cn, sn, Tt, Yt):
    Ym = ftp.fitfunc(Tt, 1, 2 * np.pi, cn, sn, 2.0, 0.0, 0.0)

    # Align the maxima of truncated and full templates
    di = np.argmax(Ym) - np.argmax(Yt)

    # Add some 'wiggle room', since maxima may be offset by 1
    Ym = [ np.array([ Ym[(j + (di + k))%len(Ym)] for j in range(len(Ym)) ]) for k in [ -1, 0, 1 ] ]

    # Align the heights of the truncated and full templates
    Ym = [ Y + (Yt[0] - Y[0]) for Y in Ym ]

    # Keep the best fit
    return Ym[np.argmin( [ rms(Y - Yt) for Y in Ym ] )]


def rms_resid_over_rms_fast(cn, sn, Tt, Yt):
    Ym = match_up_truncated_template(cn, sn, Tt, Yt)
    return rms(Yt - Ym) / rms(Yt)


def approximate_template(Tt, Yt, errfunc=rms_resid_over_rms, stop=1E-2, nharmonics=None):
    """ Fourier transforms template, returning the first H components """

    #print "fft"
    fft = np.fft.rfft(Yt)

    cn, sn = None, None
    if not nharmonics is None and int(nharmonics) > 0:
        #print "creating cn and sn"
        cn, sn = zip(*[ (p.real/len(Tt), -p.imag/len(Tt)) for i,p in enumerate(fft) \
                     if i > 0 and i <= int(nharmonics) ])

    else:

        cn, sn = zip(*[ (p.real/len(Tt), -p.imag/len(Tt)) for i,p in enumerate(fft) \
                     if i > 0 ])

        h = 1
        while errfunc(cn[:h], sn[:h], Tt, Yt) > stop:
            #print "h -> ", h
            h+=1

        cn, sn = cn[:h], sn[:h]
    return np.array(cn), np.array(sn)

normfac = lambda cn, sn : 1./np.sqrt(sum([ ss*ss + cc*cc for cc, ss in zip(cn, sn) ]))
