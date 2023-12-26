import math
import time
import numpy as np
import glog


def cal_score(a, b):
    if a == b and a != '-':
        return 1
    else:
        return -1


def print_m(items, temp, mat):
    """print score matrix or trace matrix"""
    print()
    print('     ' + '  '.join(['%3s' % i for i in temp]))
    for i, p in enumerate(items):
        line = [p] + [mat[i][j] for j in range(len(temp))]
        print('  '.join(['%3s' % i for i in line]))
    print()
    return


def traceback(items, temp, trace_mat, score_mat):
    i = len(items) - 1
    j = len(temp) - 1
    path_code = ''
    tot_score = 0

    tplt = items[1:]
    fk = temp[1:]

    while i > 0 or j > 0:
        # drct = trace_mat[i][j]
        tot_score += score_mat[i][j]
        # print(drct)
        # if drct == 0:
        #     i -= 1
        #     j -= 1
        #     path_code = '0' + path_code
        # elif drct == 1:
        #     j -= 1
        #     path_code = '1' + path_code
        # elif drct == 2:
        #     i -= 1
        #     path_code = '2' + path_code

        if i > 0 and j > 0:
            drct = [score_mat[i - 1][j - 1], score_mat[i][j - 1], score_mat[i - 1][j]]
        elif i == 0:
            drct = [-100, score_mat[i][j - 1], -100]
        elif j == 0:
            drct = [-100, -100, score_mat[i - 1][j]]
        path = drct.index(max(drct))
        if path == 0:
            path_code = '0' + path_code
            i -= 1
            j -= 1
        elif path == 1:
            path_code = '1' + path_code
            j -= 1
        elif path == 2:
            path_code = '2' + path_code
            i -= 1

    return path_code, tot_score


def pretty_print_align(items, temp, path_code):
    '''
    return pair alignment result string from
    path code: 0 for match, 1 for gap in tplt, 2 for gap in fk
    '''
    align1 = ''
    middle = ''
    align2 = ''

    tplt = items.copy()
    fk = temp.copy()
    match_fk = []
    match_tplt = []
    i = 0
    j = 0

    for p in path_code:
        if p == '0':
            align1 = align1 + ' ' + str(tplt[0]) + ' '
            align2 = align2 + ' ' + str(fk[0]) + ' '
            if tplt[0] == fk[0]:
                middle = middle + '  |  '
                match_fk.append(i)
                match_tplt.append(j)
            else:
                middle = middle + '     '
            tplt = tplt[1:]
            fk = fk[1:]
            i += 1
            j += 1
        elif p == '1':
            align1 = align1 + '  -  '
            align2 = align2 + ' ' + str(fk[0]) + ' '
            middle = middle + '     '
            fk = fk[1:]
            i += 1
        elif p == '2':
            align1 = align1 + ' ' + str(tplt[0]) + ' '
            align2 = align2 + '  -  '
            middle = middle + '     '
            tplt = tplt[1:]
            j += 1

    print('Alignment:\n\n   ' + align1 + '\n   ' + middle + '\n   ' + align2 + '\n')
    return match_fk, match_tplt


def strip_outliers(arr):
    arr_j = []
    if 3 <= len(arr) <= 15:
        Gvalue = [1.15, 1.46, 1.67, 1.82, 1.94, 2.03, 2.11, 2.18, 2.23, 2.29, 2.33, 2.37, 2.41]
        # sorted_arr = sorted(arr, key=abs)
        mean_arr = np.mean(arr)
        std_arr = np.std(arr)
        for i in range(len(arr)):
            G = abs(abs(arr[i]) - mean_arr) / std_arr
            print(arr[i], G)
            if G >= Gvalue[len(arr) - 3]:
                arr_j.append(-1)
            else:
                arr_j.append(1)
    return arr_j


def get_id_scale(intercept, template):
    interval = [(intercept[i + 1] - intercept[i]) for i in range(len(intercept) - 1)]

    count = len(interval)
    template_len = len(template)

    template_used = template * (math.ceil(count / template_len))
    template_used.extend(template[:count-1])

    out_score = -100
    out_code = ''
    out_fk = []
    mean_dst = 100
    fk_idx = []
    tplt_idx = []

    for val in interval:
        if val == 0:
            continue
        for i in range(len(template)):
            # ratio = template[i] / val
            fake_tplt = np.round(np.array(interval) * template[i] / val)
            fake_dist = []
            for j in range(len(fake_tplt)):
                fk = fake_tplt[j]
                dist = np.abs(np.array(template) / fk - 1)
                d_v = np.min(dist)
                if d_v > 0.1:
                    fake_tplt[j] = -1
                else:
                    try:
                        fake_tplt[j] = template[list(dist).index(d_v)]
                    except:
                        fake_tplt[j] = -1
                fake_dist.append(d_v)

            items = template_used[i:i + count]
            items.insert(0, '-')
            temp = list(np.int_(fake_tplt))
            temp.insert(0, '-')

            score_mat = {}
            trace_mat = {}

            for j, v in enumerate(items):
                score_mat[j] = {}
                trace_mat[j] = {}
                for k, s in enumerate(temp):
                    if j == 0:
                        score_mat[j][k] = -k
                        trace_mat[j][k] = 1
                        continue
                    if k == 0:
                        score_mat[j][k] = -j
                        trace_mat[j][k] = 1
                        continue
                    ul = score_mat[j-1][k-1] + cal_score(v, s)
                    l = score_mat[j][k-1] + cal_score('-', s)
                    u = score_mat[j-1][k] + cal_score(v, '-')
                    picked = max([ul, l, u])
                    score_mat[j][k] = picked
                    trace_mat[j][k] = [ul, l, u].index(picked)

            # print_m(items, temp, score_mat)
            # print_m(items, temp, trace_mat)
            code, score = traceback(items, temp, trace_mat, score_mat)

            # if score == 7:
            #     print(items, temp)
            #     print_m(items, temp, trace_mat)
            #     pretty_print_align(template_used[i:], np.int_(fake_tplt), code)

            if score > out_score:
                # out_code = code
                # out_fk = fake_tplt
                max_val = i + len(code)
                template_match = template * (math.ceil(max_val / template_len))
                fk_idx, tplt_idx = pretty_print_align(template_match[i:i+len(code)], np.int_(fake_tplt), code)
                if len(fk_idx) == 0:
                    continue
                tplt_idx = list(np.array(tplt_idx) + i)
                sum_dst = 0
                for idx in fk_idx:
                    if idx >= 0:
                        sum_dst += fake_dist[idx]
                mean_dst = sum_dst / len(fk_idx)
                out_score = score
            elif score == out_score:
                max_val = i + len(code)
                template_match = template * (math.ceil(max_val / template_len))
                fk_idx_temp, tplt_idx_temp = pretty_print_align(template_match[i:i+len(code)], np.int_(fake_tplt), code)
                sum_dst_temp = 0
                for idx in fk_idx_temp:
                    if idx >= 0:
                        sum_dst_temp += fake_dist[idx]
                mean_dst_temp = sum_dst_temp / len(fk_idx_temp)
                if mean_dst_temp < mean_dst:
                    out_score = score
                    mean_dst = mean_dst_temp
                    fk_idx = fk_idx_temp
                    tplt_idx = list(np.array(tplt_idx_temp) + i)

            # max_val = i + len(code)
            # template_match = template * (math.ceil(max_val / template_len))
            # fk_idx_temp, tplt_idx_temp = pretty_print_align(template_match[i:i+len(code)], np.int_(fake_tplt), code)
            #
            # if len(fk_idx_temp) == 0:
            #     continue
            #
            # sum_dst_temp = 0
            # for idx in fk_idx_temp:
            #     sum_dst_temp += fake_dist[idx]
            # score = sum_dst_temp / len(fk_idx_temp)
            #
            # if score < out_score:
            #     out_score = score
            #     fk_idx = fk_idx_temp
            #     tplt_idx = tplt_idx_temp


    # start = out_code.index('0')
    # end = out_code.rfind('0')
    #
    # std_dist = np.sum(out_fk[start:end+1])
    # im_dist = intercept[end + 1] - intercept[start]

    scale_lst = []
    for i in range(len(fk_idx)):
        scale_lst.append(interval[fk_idx[i]] / template_used[tplt_idx[i]])
    scale_j = strip_outliers(scale_lst)
    fk_idx = [fk_idx[i] for i in range(len(scale_j)) if scale_j[i] > 0]
    tplt_idx = [tplt_idx[i] for i in range(len(scale_j)) if scale_j[i] > 0]

    try:
        im_dist = intercept[fk_idx[-1] + 1] - intercept[fk_idx[0]]
        std_dist = np.sum(template_used[tplt_idx[0]:tplt_idx[-1] + 1])
    except Exception as e:
        return -1, -1

    scale = im_dist / std_dist

    # exp_tplt = np.round(np.array(interval) / scale)

    index = np.zeros(len(intercept)) - 1
    for i in range(len(fk_idx)):
        j = fk_idx[i]
        if index[j] == -1:
            index[j] = tplt_idx[i]
        if index[j] > 8:
            index[j] -= 9
        index[j + 1] = tplt_idx[i] + 1
        if index[j + 1] > 8:
            index[j + 1] -= 9

    for i in range(len(index)):
        idx = index[i]
        if idx == -1:
            l = -1
            r = -1
            for j in range(i):
                _i = i - j
                if index[_i] >= 0:
                    l = int(index[_i])
                    break
            for j in range(i+1, len(index)):
                i_ = j
                if index[i_] >= 0:
                    r = int(index[i_])
                    break

            if l != r:
                if l >= 0:
                    d_l = round(interval[i - 1] / scale)
                    for n in range(1, 9):
                        if abs(d_l - np.sum(template_used[l:l + n])) <= 3:
                            index[i] = l + n
                            if l + n > 8:
                                index[i] -= 9
                            break

                if r >= 0:
                    d_r = round(interval[i] / scale)
                    for n in range(1, 9):
                        if abs(d_r - np.sum(template_used[r - n + 9:r + 9])) <= 3:
                            index[i] = r - n
                            if r - n < 0:
                               index[i] += 9
                            break

            for k in range(len(index)):
                if index[k] == index[i]:
                    if abs(k - i) <= 2:
                        index[i] = -1
                        break

    return np.int_(index), scale


def intersect(s1, s2, s_idx, template, tolerance=15):
    # 计算匹配程度
    o_itcp = []
    o_idx = []
    tp_tmp = [template[idx] for idx in s_idx]
    i = j = 0
    st = 0
    ed = 0
    err_dist = 0
    while i < len(s1) and j < len(s2):
        if abs(s1[i] - s2[j]) <= tolerance:
            err_dist += abs(s1[i] - s2[j])
            # 如果实际截距和理论截距在容忍范围内认为匹配
            if len(o_itcp) == 0:
                st = i
            o_itcp.append(s2[j])
            o_idx.append(s_idx[i])
            i += 1
            j += 1
            ed = i
        elif s1[i] - s2[j] < 0:
            i += 1
        else:
            # 实际截距中无法匹配的值index置为-1
            o_idx.append(-1)
            j += 1

    if len(o_idx) < len(s2):
        for k in range(len(s2) - len(o_idx)):
            o_idx.append(-1)

    # 通过有效index计算标准模板的总宽度
    if st == ed:
        tp_dist = 0
    else:
        tp_dist = np.sum(tp_tmp[st:ed-1])

    return o_itcp, o_idx, tp_dist, err_dist


def map_by_intercept(intercept, template, length=2040):
    # 根据index获取模板宽度
    def get_template(idx):
        if idx < 0 or idx >= len(template):
            return template[idx % len(template)]
        else:
            return template[idx]

    # 通过实际截距计算实际宽度
    interval = [(intercept[i + 1] - intercept[i]) for i in range(len(intercept) - 1)]
    # count = len(interval)
    # template_len = len(template)

    # template_used = template * ((math.ceil(count / template_len)) + 2)
    # template_used.extend(template[:count-1])

    index = []
    eff_itcp = []
    tp_dist = 0
    o_score = 0
    err_min = 0
    # 用每一个实际截距和每一个模板宽度进行枚举计算
    for i in range(len(interval)):
        val = interval[i]
        itcp = intercept[i]
        if itcp < 0:
            continue
        for j in range(len(template)):
            fake_itcp = [itcp]
            fake_idx = [j]

            # 计算临时scale
            scale_tmp = val / template[j]
            if scale_tmp < 0.5 or scale_tmp > 1.8:
                continue

            # 推算该临时scale下的理论截距列表和对应的index列表
            k = j - 1
            st = itcp - get_template(k) * scale_tmp
            while st >= 0:
                fake_itcp.insert(0, st)
                fake_idx.insert(0, k)
                k -= 1
                st -= get_template(k) * scale_tmp

            k = j
            st = itcp + get_template(k) * scale_tmp
            while st < length:
                fake_itcp.append(st)
                fake_idx.append(k + 1)
                k += 1
                st += get_template(k) * scale_tmp

            for n in range(len(fake_idx)):
                fake_idx[n] = fake_idx[n] % len(template)

            # 计算和判断匹配程度
            o_itcp, o_idx, tmp_dist, err_dist = intersect(fake_itcp, intercept, fake_idx, template)
            if len(o_itcp) > o_score:
                o_score = len(o_itcp)
                index = o_idx
                eff_itcp = o_itcp
                tp_dist = tmp_dist
                err_min = err_dist
            elif len(o_itcp) == o_score:
                if err_dist < err_min:
                    o_score = len(o_itcp)
                    index = o_idx
                    eff_itcp = o_itcp
                    tp_dist = tmp_dist
                    err_min = err_dist

    # 结果过滤和异常处理
    if o_score <= 2:
        return -1, -1
    if tp_dist == 0:
        return -1, -1

    # 计算scale
    im_dist = eff_itcp[-1] - eff_itcp[0]

    scale = im_dist / tp_dist

    return index, scale







if __name__ == '__main__':
    intercept = [8.59643435980551, 300.7980075830793, 451.0, 818.2894456289979, 1096.2808544303798, 1454.8434579439252, 1843.042372881356]
    # interval = [299.3347758062856, 254.15549450549446, 15.879411764705878, 213.68797953964201, 184.2683229813664, 16.978571428571513, 293.46502821502804, 196.57068607068618, 229.21428571428578]
    template = [240, 300, 330, 390, 390, 330, 300, 240, 420]

    # get_id_scale(intercept, template)
    map_by_intercept(intercept, template)

