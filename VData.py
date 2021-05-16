class ImageDataSet():

    def __init__(self, filepathes, label_dir, conf_threshold = 0.35):

        filepathes.sort()
        
        labels = os.listdir(label_dir)
        labels.sort()

        self.filepathes = filepathes
        self.labels = []
        self.conf_threshold = conf_threshold
        self.shapes = []

        for i, filename in enumerate(labels):

            img = Image.open(filepathes[i])
            img_width, img_height = img.size

            temp_list = list()

            with open(os.path.join(label_dir, filename), mode='r') as f:
                lines = f.readlines()
                for line in lines:

                    temp_dict = dict()

                    # read yolov5 output format
                    lst = [float(a) for a in line.strip('\n').split(' ')]
                    label, x_center_norm, y_center_norm, width_norm, height_norm, predict_conf = lst

                    x_center = int(x_center_norm * img_width)
                    y_center = int(y_center_norm * img_height)
                    box_width = int(width_norm * img_width)
                    box_height = int(height_norm * img_height)

                    temp_dict['label'] = 'S' if label == 0 else 'others'
                    temp_dict['x_center'] = x_center
                    temp_dict['y_center'] = y_center
                    temp_dict['box_width'] = box_width
                    temp_dict['box_height'] = box_height
                    temp_dict['conf'] = predict_conf

                    temp_list.append(temp_dict)

            temp_list.sort(key=lambda x: x.get('y_center'))
            self.labels.append(temp_list)
    
    def __len__(self):

        return len(self.filepathes)

    def __getitem__(self, idx):

        return Image.open(filepathes[idx]), labels[idx]
    
    def plotDataset(self, idxs = None):

        getLineStyle = {
            'S': '-g',
            'others': '-y',
            'screw': '-r',
        }

        if idxs is None:
            idxs = list(range(len(self.filepathes)))
        elif type(idxs) is int:
            idxs = [idxs]

        for idx in idxs:
            
            img = Image.open(self.filepathes[idx])
            label = self.labels[idx]

            plt.figure(figsize=(10,10))
            plt.imshow(img, cmap='gray')

            isFirst = True

            for L in label:

                if L['conf'] < self.conf_threshold:
                    continue

                if isFirst == False:
                    plt.plot([x_c, L['x_center']], [y_c, L['y_center']], '-c')
                    isFirst = False
                else:
                    isFirst = False

                x_c, y_c = L['x_center'], L['y_center']
                w, h = L['box_width'], L['box_height']

                x_min, y_min, x_max, y_max = x_c-w/2, y_c-h/2, x_c+w/2, y_c+h/2

                style = getLineStyle[L['label']]
            
                plt.plot([x_min, x_min], [y_min, y_max], style)
                plt.plot([x_max, x_max], [y_min, y_max], style)
                plt.plot([x_min, x_max], [y_min, y_min], style)
                plt.plot([x_min, x_max], [y_max, y_max], style)

                if L['label'] != 'S':
                    plt.text(x_c, y_c, L['conf'])
                    plt.plot(x_c, y_c, 'r+')

    def detectShape(self, model):

        my_transformer = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])

        for idx, filepath in enumerate(self.filepathes):

            img = Image.open(filepath)

            for L in self.labels[idx]:
                if L['label'] != 'S' and L['conf'] > self.conf_threshold:

                    x_c, y_c = L['x_center'], L['y_center']
                    w, h = L['box_width'], L['box_height']

                    box = (x_c-w/2, y_c-h/2, x_c+w/2, y_c+h/2)
                    img_box = img.crop(box)
                    input_img = my_transformer(img_box)
                    input_img = torch.unsqueeze(input_img, 0)

                    model.eval()
                    with torch.no_grad():
                        output = model(input_img)
                        prob = torch.sigmoid(output)
                        prob = TF.resize(prob, [*(img_box.size)][::-1])
                        outimg = prob.squeeze().detach().numpy()
                        masked = np.ma.masked_where(outimg < 0.8, outimg)
                    plt.figure(figsize=(5,5))
                    plt.imshow(img_box, cmap='gray')
                    plt.imshow(masked, cmap='jet', alpha=0.3)
                    plt.savefig(os.path.join('./testmasked/', str(idx)+L['label']))
                    plt.close()


    def detectScrew(self, model):

        my_transformer = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])

        for idx, filepath in enumerate(self.filepathes):

            img = Image.open(filepath)
            
            for L in self.labels[idx]:
                if L['label'] != 'S':
                    
                    x_c, y_c = L['x_center'], L['y_center']
                    w, h = L['box_width'], L['box_height']

                    box = (x_c-w/2, y_c-h/2, x_c+w/2, y_c+h/2)
                    img_box = img.crop(box)
                    img_box = my_transformer(img_box)
                    img_box = torch.unsqueeze(img_box, 0)

                    model.eval()
                    with torch.no_grad():
                        outputs = model(img_box)
                        _, predicted = torch.max(outputs, 1)
                        if predicted[0] == 1:
                            L['label'] = 'screw'