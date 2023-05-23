//
//  SendView.swift
//  Tridify
//
//  Created by Maged Alosali on 21/05/2023.
//

import SwiftUI

struct SendView: View {
    var body: some View {
        VStack{
            Spacer()
            VStack (alignment: .leading){
                Text ("Images (100)")
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack {
                        ForEach(0..<100, id: \.self){_ in
                            Color.red
                                .frame(width: 200,height: 200)
                        }
                    }
                }
            }
            Spacer()
            
            VStack {
                Text("Size: ")
                    .font(.headline)
                    .foregroundColor(Color(uiColor: UIColor.lightGray))
                Button ("UPLOAD & PROCESS"){
                    
                }
                .foregroundColor(.black.opacity(98))
                .frame(width: 200, height: 50)
                .background(.white.opacity(88))
                .clipShape(Capsule())
                
            }
        }
        .padding()
    }
}

struct SendView_Previews: PreviewProvider {
    static var previews: some View {
        SendView()
            .preferredColorScheme(.dark)
    }
}
