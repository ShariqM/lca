x_num = 59
y_num = 98
log_x = np.zeros(0)
log_y = np.zeros(0)
batch_size = 10
colors = np.linspace(0, 1, batch_size)
for i in range(batch_size):
log_x = activity_log[x_num, i, :]
log_y = activity_log[y_num, i, :]
colors = np.linspace(0, 1, len(log_x))
color = COLORS[COLORS.keys()[i]]
plt.scatter(log_x, log_y, s=8, c=color, label=str(i), cmap=plt.get_cmap('autumn'), lw=0)

plt.xlabel("%d Activity" % x_num)
plt.ylabel("%d Activity" % y_num)
plt.legend(loc=4, prop={'size':6})
plt.show(block=True)
plt.savefig('plots/%s.png' % datetime.now())





top_CC_AC = max(max(CC),max(AC),top_CC_AC) * 1.1
plt.subplot(232)
act_a = np.average(activity_log[29 -1, :, :], axis=0)
act_b = np.average(activity_log[434 -1, :, :], axis=0)
#act_c = np.average(activity_log[288 -1, :, :], axis=0)
plt.xlabel('Time (steps)', fontdict={'fontsize':12})
plt.ylabel('Activity', fontdict={'fontsize':12})
plt.axis([0, num_frames, 0, max(max(act_a), max(act_b),)])
#plt.axis([0, num_frames, 0, max(max(act_a), max(act_b), max(act_c))])
plt.plot(range(num_frames), act_a, color=rcolor[run])
plt.plot(range(num_frames), act_b, color=COLORS['green'])
#plt.plot(range(num_frames), act_c, color=COLORS['blue'])
