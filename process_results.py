import matplotlib.pyplot as plt

results = ["trial1_out.txt", "trial2_out.txt", "trial3_out.txt", "trial4_out.txt", "trial5_out.txt", "trial6_out.txt", "trial7_out.txt", "trial8_out.txt", "baseline_out.txt"]

epoch = [i+1 for i in range(100)]

train_acc = []
train_loss = []
valid_acc = []
valid_loss = []

for result in results:

    with open(result, 'r') as f:
        lines = f.readlines()

        train_acc_acc = []
        train_loss_acc = []
        valid_acc_acc = []
        valid_loss_acc = []

        for line in lines:

            # for training lines
            if "- ETA:" in line:
                pass

            # for validation lines
            elif "50000/50000 [" in line:
                t_acc = float(line.split('-')[3].split(':')[1].strip())
                t_loss = float(line.split('-')[2].split(':')[1].strip())

                train_acc_acc.append(t_acc)
                train_loss_acc.append(t_loss)

                v_acc = float(line.split('-')[5].split(':')[1].strip())
                v_loss = float(line.split('-')[4].split(':')[1].strip())

                valid_acc_acc.append(v_acc)
                valid_loss_acc.append(v_loss)

        train_acc.append(train_acc_acc)
        train_loss.append(train_loss_acc)
        valid_acc.append(valid_acc_acc)
        valid_loss.append(valid_loss_acc)

train_steps = []

for j in range(len(results)):
    train_steps.append([i for i in range(len(train_loss[j]))])

plt.plot(epoch, train_loss[0], epoch, train_loss[1], epoch, train_loss[2], epoch, train_loss[3], epoch, train_loss[4], epoch, train_loss[5], epoch, train_loss[6], epoch, train_loss[7], epoch, train_loss[8])
plt.show()

plt.plot(epoch, train_acc[0], epoch, train_acc[1], epoch, train_acc[2], epoch, train_acc[3], epoch, train_acc[4], epoch, train_acc[5], epoch, train_acc[6], epoch, train_acc[7], epoch, train_acc[8])
plt.show()

plt.plot(epoch, valid_loss[0], epoch, valid_loss[1], epoch, valid_loss[2], epoch, valid_loss[3], epoch, valid_loss[4], epoch, valid_loss[5], epoch, valid_loss[6], epoch, valid_loss[7], epoch, valid_loss[8])
plt.show()

plt.plot(epoch, valid_acc[0], epoch, valid_acc[1], epoch, valid_acc[2], epoch, valid_acc[3], epoch, valid_acc[4], epoch, valid_acc[5], epoch, valid_acc[6], epoch, valid_acc[7], epoch, valid_acc[8])
plt.show()
