//
//  SendView.swift
//  Tridify
//
//  Created by Maged Alosali on 21/05/2023.
//

import SwiftUI

struct SendView: View {
    @State private var modelName = ""
    @State private var showAlert = false
    @Binding private var imageSize: Double
    private let images: [Data]
    var body: some View {
        VStack{
            Spacer()
            VStack(alignment: .leading) {
                Text("Model Name ")
                TextField("Enter your model Name", text: $modelName)
                    .padding(.horizontal, 20)
                    .font(.title3)
                    .overlay(content: {
                        RoundedRectangle(cornerRadius: 10)
                            .stroke()
                            .frame(height: 35)
                            .foregroundColor(.gray)
                    })
            }
            Spacer()
            
            VStack (alignment: .leading){
                Text ("Images (\(images.count))")
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 20) {
                        ForEach(0..<images.count, id: \.self){i in
                            Image(uiImage: UIImage(data:images[i])!)
                                .resizable()
                                .scaledToFill()
                                .frame(width: 200,height: 200)
                                
                        }
                    }
                }
            }
            Spacer()
            Spacer()
            VStack {
                Text("Size: \(String(format:"%0.2f",imageSize)) MB")
                    .font(.headline)
                    .foregroundColor(Color(uiColor: UIColor.lightGray))
                Button ("UPLOAD & PROCESS"){
                    let check = modelName.trimmingCharacters(in: .whitespaces)
                    if (check.isEmpty) {
                        showAlert.toggle()
                    }
                }
                .foregroundColor(.black.opacity(98))
                .frame(width: 200, height: 50)
                .background(.white.opacity(88))
                .clipShape(Capsule())
                
            }
        }
        .padding()
        .alert("Empty Model Name", isPresented: $showAlert) {
            Button("Ok", role: .cancel){}
        } message: {
            Text("Please enter a name for the model")
        }
    }
    init (imageSize: Binding<Double>, images: [Data]){
        _imageSize = imageSize
        self.images = images
    }
}


